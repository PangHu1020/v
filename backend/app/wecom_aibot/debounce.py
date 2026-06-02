"""Channel-layer per-conversation 500ms debounce.

Customer IM clients sometimes split a thought across several consecutive
messages within a few hundred milliseconds (``"hi"`` -> ``"i have a"`` ->
``"question about order 123"``). Letting each fragment trigger an
independent agent turn wastes tokens and produces choppy replies.

The debouncer collapses bursts: messages from the same
``(channel, channel_user_id)`` arriving within ``window_ms`` of each other
are accumulated, and only the last in the burst (carrying concatenated text)
proceeds to the bus.

Implementation notes
--------------------
- A Redis hash ``debounce:<channel>:<user>`` holds the pending merged text
  + arrival timestamp + dedup key.
- ``observe()`` updates that hash and schedules an asyncio task to flush
  it after ``window_ms`` of inactivity.
- The flush task wakes up every ``window_ms`` and checks whether the
  pending entry is still the latest; if so it pops it and yields a final
  :class:`SystemMessage`. If a newer message has bumped the timestamp the
  flush re-arms.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

import redis.asyncio as redis_async

from backend.app.bus.messages import SystemMessage
from backend.v.utils.logging import get_logger

_log = get_logger("channels.debounce")

_HASH_PREFIX = "debounce"

DispatchFn = Callable[[SystemMessage], Awaitable[None]]


class Debouncer:
    """Per-identity message debouncer backed by Redis hashes + asyncio timers."""

    def __init__(
        self,
        client: redis_async.Redis,
        *,
        window_ms: int,
        dispatch: DispatchFn,
    ) -> None:
        self._client = client
        self._window = window_ms / 1000.0
        self._dispatch = dispatch
        self._timers: dict[str, asyncio.Task[None]] = {}

    @staticmethod
    def _key(channel: str, channel_user_id: str) -> str:
        return f"{_HASH_PREFIX}:{channel}:{channel_user_id}"

    async def observe(self, message: SystemMessage) -> None:
        """Record a fresh inbound message and (re)arm the flush timer.

        Args:
            message: The post-decryption normalized message; may be partial
                if it represents one fragment of a typing burst.
        """
        key = self._key(message.channel, message.channel_user_id)

        # Merge with any pending fragment under the same key.
        prior = await self._client.hgetall(key)
        prior_text = prior.get(b"text", b"").decode("utf-8") if prior else ""
        prior_dedup = prior.get(b"dedup_key", b"").decode("utf-8") if prior else ""
        merged_text = f"{prior_text}\n{message.text}".strip() if prior_text else message.text
        merged_dedup = f"{prior_dedup}|{message.dedup_key}" if prior_dedup else message.dedup_key

        await self._client.hset(
            key,
            mapping={
                b"text": merged_text.encode("utf-8"),
                b"dedup_key": merged_dedup.encode("utf-8"),
                b"channel": message.channel.encode("utf-8"),
                b"channel_user_id": message.channel_user_id.encode("utf-8"),
                b"received_at": message.received_at.isoformat().encode("utf-8"),
                b"attachments": json.dumps(message.attachments).encode("utf-8"),
            },
        )
        await self._client.pexpire(key, int(self._window * 1000) * 4)

        timer = self._timers.get(key)
        if timer and not timer.done():
            timer.cancel()
        self._timers[key] = asyncio.create_task(self._flush_after_window(key))

    async def _flush_after_window(self, key: str) -> None:
        try:
            await asyncio.sleep(self._window)
        except asyncio.CancelledError:
            return

        pending = await self._client.hgetall(key)
        if not pending:
            return

        await self._client.delete(key)
        try:
            merged = SystemMessage(
                channel=pending[b"channel"].decode("utf-8"),  # type: ignore[arg-type]
                channel_user_id=pending[b"channel_user_id"].decode("utf-8"),
                text=pending[b"text"].decode("utf-8"),
                attachments=json.loads(pending[b"attachments"]),
                dedup_key=pending[b"dedup_key"].decode("utf-8"),
            )
        except Exception as exc:
            _log.error("debounce.deserialize_failed", key=key, error=str(exc))
            return

        try:
            await self._dispatch(merged)
        except Exception as exc:
            _log.error("debounce.dispatch_failed", key=key, error=str(exc))

    async def shutdown(self) -> None:
        """Cancel all pending timers; in-flight dispatches finish."""
        for t in self._timers.values():
            if not t.done():
                t.cancel()
        self._timers.clear()
