"""WeCom 智能机器人 WebSocket client.

Wire protocol notes
-------------------

The WeCom 智能机器人 platform speaks a ``cmd``-keyed JSON protocol over
WebSocket, NOT a generic ``type``-keyed one. Auth is performed by
posting an ``aibot_subscribe`` frame after the socket opens — there is
no HTTP ``Authorization`` header.

Frame summary (all JSON-over-text):

==========================  ===============================================
``cmd``                     Direction / purpose
==========================  ===============================================
``aibot_subscribe``         out — initial auth handshake (bot_id + secret)
``aibot_msg_callback``      in  — inbound user message
``aibot_send_msg``          out — proactive message (no req_id)
``aibot_respond_msg``       out — direct reply to a callback (uses req_id)
``ping``                    both — heartbeat
==========================  ===============================================

Inbound ``aibot_msg_callback`` body shape::

    {
        "cmd": "aibot_msg_callback",
        "header": {"req_id": "..."},
        "body": {
            "msgid": "...",
            "from": {"userid": "..."},
            "chatid": "...",            # may equal userid in 1-on-1
            "msgtype": "text" | "mixed" | "image" | ...,
            "text":  {"content": "..."},      # for text
            "mixed": {"items": [...]}         # for mixed (text + media)
        }
    }

We map ``chatid`` (falling back to ``from.userid``) onto our internal
``channel_user_id``: it is the conversation key the bus shards by, and
in 1-on-1 chats it is effectively the user. ``msgid`` becomes the
``dedup_key``. Non-text msgtypes are dropped for now (multimodal is a
later ticket).

For replies, we cache the most recent ``req_id`` per chat so
:meth:`send_text` can issue a routed ``aibot_respond_msg`` when one is
known, and a proactive ``aibot_send_msg`` otherwise.
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
from backend.app.wecom_aibot.debounce import Debouncer
from backend.v.utils.logging import get_logger

_log = get_logger("channels.wecom_aibot.client")

_BACKOFF_INITIAL = 1.0
_BACKOFF_MAX = 60.0
_MAX_MESSAGE_LEN = 4000

# WeCom AI Bot protocol cmd constants — see module docstring.
_CMD_SUBSCRIBE = "aibot_subscribe"
_CMD_CALLBACK = "aibot_msg_callback"
_CMD_SEND = "aibot_send_msg"
_CMD_RESPOND = "aibot_respond_msg"
_CMD_PING = "ping"


class WecomAibotClient:
    """Persistent WebSocket client for WeCom 智能机器人."""

    def __init__(
        self,
        *,
        ws_url: str,
        bot_id: str,
        secret: str,
        heartbeat_seconds: int,
        debouncer: Debouncer,
    ) -> None:
        self._ws_url = ws_url
        self._bot_id = bot_id
        self._secret = secret
        self._heartbeat = heartbeat_seconds
        self._debouncer = debouncer
        self._ws: ClientConnection | None = None
        self._send_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        # chat_id -> most recent req_id, for routing aibot_respond_msg.
        self._last_req_ids: dict[str, str] = {}

    async def run(self) -> None:
        """Main loop: connect, handshake, pump frames, reconnect on drop."""
        backoff = _BACKOFF_INITIAL
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=None,  # we drive heartbeats explicitly
                ) as ws:
                    self._ws = ws
                    backoff = _BACKOFF_INITIAL
                    _log.info("wecom_aibot.ws.connected")
                    await self._subscribe()
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
            except TimeoutError:
                pass
            backoff = min(backoff * 2, _BACKOFF_MAX)

    async def stop(self) -> None:
        """Signal the run loop to exit and close the current socket."""
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                await ws.close()
            except Exception:  # noqa: S110 — shutting down, swallow close errors
                pass

    async def send_text(self, channel_user_id: str, text: str) -> None:
        """Send a text reply to the chat identified by ``channel_user_id``.

        ``channel_user_id`` is the WeCom ``chatid`` we stored on the way
        in. If a recent ``req_id`` is cached for that chat we issue a
        routed ``aibot_respond_msg`` (the platform threads the reply to
        the originating message) **and consume the req_id** — only the
        first reply to a given inbound callback can use ``respond``.
        Subsequent replies (e.g., the 2nd/3rd segment of a multi-bubble
        answer) fall through to a proactive ``aibot_send_msg``.

        Reusing the same ``req_id`` across multiple ``respond`` frames
        causes the platform to process them in parallel, breaking the
        ordering the customer sees.

        Drops silently if no socket is currently connected.
        """
        ws = self._ws
        if ws is None:
            _log.warning("wecom_aibot.send.no_socket", channel_user_id=channel_user_id)
            return

        content = text[:_MAX_MESSAGE_LEN]
        req_id = self._last_req_ids.pop(channel_user_id, None)
        if req_id:
            payload: dict[str, Any] = {
                "cmd": _CMD_RESPOND,
                "header": {"req_id": req_id},
                "body": {
                    "msgtype": "markdown",
                    "markdown": {"content": content},
                },
            }
        else:
            payload = {
                "cmd": _CMD_SEND,
                "body": {
                    "chatid": channel_user_id,
                    "msgtype": "markdown",
                    "markdown": {"content": content},
                },
            }
        async with self._send_lock:
            await ws.send(json.dumps(payload, ensure_ascii=False))
        _log.info(
            "wecom_aibot.outbound.sent",
            channel_user_id=channel_user_id,
            mode="respond" if req_id else "send",
            len=len(content),
        )

    async def _subscribe(self) -> None:
        ws = self._ws
        assert ws is not None
        payload = {
            "cmd": _CMD_SUBSCRIBE,
            "body": {"bot_id": self._bot_id, "secret": self._secret},
        }
        async with self._send_lock:
            await ws.send(json.dumps(payload))
        _log.info("wecom_aibot.subscribe.sent", bot_id=self._bot_id)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._heartbeat)
            ws = self._ws
            if ws is None:
                return
            try:
                async with self._send_lock:
                    await ws.send(json.dumps({"cmd": _CMD_PING, "ts": int(time.time())}))
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
            if not isinstance(frame, dict):
                continue
            await self._handle_frame(frame)

    async def _handle_frame(self, frame: dict[str, Any]) -> None:
        cmd = frame.get("cmd")
        if cmd == _CMD_PING:
            # Server-initiated ping — reply in kind so the connection
            # stays open even if our heartbeat is mid-sleep.
            ws = self._ws
            if ws is not None:
                async with self._send_lock:
                    await ws.send(json.dumps({"cmd": _CMD_PING, "ts": int(time.time())}))
            return
        if cmd != _CMD_CALLBACK:
            # subscribe-ack, send-ack, and unknown cmds — log + drop.
            _log.info("wecom_aibot.frame.skip", cmd=cmd)
            return

        body = frame.get("body")
        if not isinstance(body, dict):
            _log.warning("wecom_aibot.callback.bad_body")
            return

        msg_id = str(body.get("msgid") or "").strip()
        sender = body.get("from") or {}
        user_id = str(sender.get("userid") or "").strip() if isinstance(sender, dict) else ""
        chat_id = str(body.get("chatid") or user_id).strip()
        if not chat_id:
            _log.warning("wecom_aibot.callback.no_chat_id", msg_id=msg_id)
            return

        # Cache req_id so a later send_text can route the reply.
        header = frame.get("header")
        if isinstance(header, dict):
            req_id = str(header.get("req_id") or "").strip()
            if req_id:
                self._last_req_ids[chat_id] = req_id

        text = _extract_text(body)
        if not text:
            _log.info("wecom_aibot.callback.empty_text", msg_id=msg_id, chat_id=chat_id)
            return

        if not msg_id:
            msg_id = f"wecom_aibot-{chat_id}-{int(time.time() * 1000)}"

        sys_msg = SystemMessage(
            channel="wecom_aibot",
            channel_user_id=chat_id,
            text=text,
            dedup_key=msg_id,
            received_at=datetime.now(UTC),
        )
        await self._debouncer.observe(sys_msg)
        _log.info(
            "wecom_aibot.event.observed",
            dedup_key=msg_id,
            chat_id=chat_id,
            user_id=user_id,
        )


def _extract_text(body: dict[str, Any]) -> str:
    """Pull the plain-text portion out of a callback ``body``.

    Supports ``msgtype="text"`` and ``msgtype="mixed"`` (concatenates
    every text item). Other msgtypes (image, voice, video, file) return
    an empty string — multimodal is a later ticket.
    """
    msg_type = str(body.get("msgtype") or "")
    if msg_type == "text":
        text_obj = body.get("text") or {}
        if isinstance(text_obj, dict):
            return str(text_obj.get("content") or "").strip()
        return ""
    if msg_type == "mixed":
        mixed = body.get("mixed") or {}
        items = mixed.get("items") if isinstance(mixed, dict) else None
        if not isinstance(items, list):
            return ""
        parts: list[str] = []
        for item in items:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text") or {}
                if isinstance(t, dict):
                    parts.append(str(t.get("content") or ""))
        return " ".join(parts).strip()
    return ""
