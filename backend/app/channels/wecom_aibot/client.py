"""WeCom 智能机器人 WebSocket client.

Responsibilities:

- Maintain a single long-lived WebSocket to the WeCom 智能机器人 endpoint,
  reconnecting with exponential backoff on drop.
- Send periodic JSON heartbeats so the server-side does not idle the
  socket out.
- Decode inbound JSON frames into :class:`SystemMessage` (channel
  ``wecom_aibot``) and hand them to the channel-layer
  :class:`Debouncer`.
- Expose ``send_text`` so the owning worker process can forward outbound
  replies received from the Redis pub/sub publisher.

The wire format here is intentionally generic JSON-over-WS; WeCom's
智能机器人 protocol differs by tenant and is configured via
``WECOM_AIBOT_*`` env vars. Each frame is expected to look like::

    {
        "type": "message",
        "from": "<external_user_id>",
        "text": "<content>",
        "msg_id": "<stable id>"
    }

Unknown ``type`` values are logged and dropped; ``ping`` / ``pong`` are
handled inline.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection

from backend.app.bus.messages import SystemMessage
from backend.app.channels.debounce import Debouncer
from backend.v.utils.logging import get_logger

_log = get_logger("channels.wecom_aibot.client")

_BACKOFF_INITIAL = 1.0
_BACKOFF_MAX = 60.0


class WecomAibotClient:
    """Persistent WebSocket client for WeCom 智能机器人."""

    def __init__(
        self,
        *,
        ws_url: str,
        token: str,
        heartbeat_seconds: int,
        debouncer: Debouncer,
    ) -> None:
        self._ws_url = ws_url
        self._token = token
        self._heartbeat = heartbeat_seconds
        self._debouncer = debouncer
        self._ws: ClientConnection | None = None
        self._send_lock = asyncio.Lock()
        self._stop = asyncio.Event()

    async def run(self) -> None:
        """Main loop: connect, pump frames, reconnect on drop."""
        backoff = _BACKOFF_INITIAL
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self._ws_url,
                    additional_headers={"Authorization": f"Bearer {self._token}"}
                    if self._token
                    else None,
                    ping_interval=None,  # we drive heartbeats explicitly
                ) as ws:
                    self._ws = ws
                    backoff = _BACKOFF_INITIAL
                    _log.info("wecom_aibot.ws.connected")
                    heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                    try:
                        await self._read_loop()
                    finally:
                        heartbeat_task.cancel()
                        try:
                            await heartbeat_task
                        except asyncio.CancelledError:
                            pass
                        self._ws = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log.warning(
                    "wecom_aibot.ws.disconnected",
                    error=str(exc),
                    backoff=backoff,
                )

            if self._stop.is_set():
                break
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                break  # stop event fired during the backoff sleep
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, _BACKOFF_MAX)

    async def stop(self) -> None:
        """Signal the run loop to exit and close the current socket."""
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def send_text(self, channel_user_id: str, text: str) -> None:
        """Send a text reply over the current WS to ``channel_user_id``.

        Drops silently if no socket is currently connected; the caller
        is the pub/sub forwarder, and re-publishing while disconnected
        is the publisher's problem (or the user's, on long outages).
        """
        ws = self._ws
        if ws is None:
            _log.warning("wecom_aibot.send.no_socket", channel_user_id=channel_user_id)
            return
        frame = {
            "type": "message",
            "to": channel_user_id,
            "text": text,
        }
        async with self._send_lock:
            await ws.send(json.dumps(frame, ensure_ascii=False))
        _log.info("wecom_aibot.outbound.sent", channel_user_id=channel_user_id, len=len(text))

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._heartbeat)
            ws = self._ws
            if ws is None:
                return
            try:
                async with self._send_lock:
                    await ws.send(json.dumps({"type": "ping", "ts": int(time.time())}))
            except Exception as exc:
                _log.warning("wecom_aibot.heartbeat.failed", error=str(exc))
                return

    async def _read_loop(self) -> None:
        ws = self._ws
        assert ws is not None
        async for raw in ws:
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError as exc:
                _log.warning("wecom_aibot.frame.bad_json", error=str(exc))
                continue
            await self._handle_frame(frame)

    async def _handle_frame(self, frame: dict[str, Any]) -> None:
        ftype = frame.get("type")
        if ftype in ("pong", "ack"):
            return
        if ftype == "ping":
            ws = self._ws
            if ws is not None:
                async with self._send_lock:
                    await ws.send(json.dumps({"type": "pong", "ts": int(time.time())}))
            return
        if ftype != "message":
            _log.info("wecom_aibot.frame.skip", type=ftype)
            return

        from_user = frame.get("from") or ""
        text = frame.get("text") or ""
        msg_id = frame.get("msg_id") or f"wecom_aibot-{from_user}-{int(time.time() * 1000)}"
        if not from_user or not text:
            _log.warning("wecom_aibot.frame.missing_fields", frame=frame)
            return

        sys_msg = SystemMessage(
            channel="wecom_aibot",
            channel_user_id=from_user,
            text=text,
            dedup_key=msg_id,
            received_at=datetime.now(UTC),
        )
        await self._debouncer.observe(sys_msg)
        _log.info("wecom_aibot.event.observed", dedup_key=msg_id)
