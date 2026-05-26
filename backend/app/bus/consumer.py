"""Bus consumer: one task per shard, strict per-shard serial processing.

Per-shard XREADGROUP with ``count=1`` and ``block_ms`` blocks until either an
entry arrives or the timeout expires. Failures route to the shared DLQ stream
(``<prefix>:dlq``) and the original entry is acknowledged so the consumer
group does not redeliver. Group-F's worker will install a real handler that
runs the LangGraph turn; here the consumer is generic and accepts any
``handler: Callable[[SystemMessage], Awaitable[None]]``.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Awaitable, Callable

from backend.app.bus.messages import SystemMessage
from backend.app.bus.shard import RedisStreamShard
from backend.v.utils.logging import bind_request, get_logger

Handler = Callable[[SystemMessage], Awaitable[None]]

_log = get_logger("bus.consumer")


class BusConsumer:
    """Subscribe to all shard streams and dispatch each entry to ``handler``."""

    def __init__(
        self,
        shard: RedisStreamShard,
        *,
        group: str,
        consumer_name: str | None = None,
        block_ms: int = 5000,
    ) -> None:
        self._shard = shard
        self._group = group
        self._consumer_name = consumer_name or f"{socket.gethostname()}:{id(self)}"
        self._block_ms = block_ms
        self._stop_event = asyncio.Event()

    async def stop(self) -> None:
        """Request graceful shutdown. In-flight handler invocations finish first."""
        self._stop_event.set()

    async def run(self, handler: Handler) -> None:
        """Spawn one task per shard and block until ``stop()`` is called.

        Each shard task strictly serializes its own stream: one entry at a
        time, processed and acked before the next is read. Distinct shards
        run concurrently as separate asyncio tasks.
        """
        for key in self._shard.all_stream_keys():
            await self._shard.ensure_group(key, self._group)

        tasks = [
            asyncio.create_task(self._run_shard(key, handler))
            for key in self._shard.all_stream_keys()
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    async def _run_shard(self, key: str, handler: Handler) -> None:
        while not self._stop_event.is_set():
            entries = await self._shard.xreadgroup(
                self._group,
                self._consumer_name,
                [key],
                count=1,
                block_ms=self._block_ms,
            )
            if not entries:
                # Real Redis already blocked for ``block_ms``. Some test
                # backends (fakeredis) return immediately, so a small sleep
                # here both avoids hot-spin and gives ``stop_event`` a chance
                # to be observed promptly.
                await asyncio.sleep(0.01)
                continue
            # entries: [(stream_key, [(msg_id, {b"json": <bytes>})])]
            for _stream_key, batch in entries:
                for msg_id, fields in batch:
                    await self._dispatch(key, msg_id, fields, handler)

    async def _dispatch(
        self,
        key: str,
        msg_id: bytes,
        fields: dict[bytes, bytes],
        handler: Handler,
    ) -> None:
        msg_id_str = msg_id.decode("ascii")
        try:
            payload = fields[b"json"]
            message = SystemMessage.model_validate_json(payload)
        except Exception as exc:
            _log.error(
                "bus.consume.deserialize_failed",
                stream=key,
                entry_id=msg_id_str,
                error=str(exc),
            )
            await self._send_to_dlq(key, msg_id_str, fields, reason=f"deserialize: {exc}")
            await self._shard.xack(key, self._group, msg_id)
            return

        with bind_request(
            request_id=msg_id_str,
            channel=message.channel,
            channel_user_id=message.channel_user_id,
        ):
            try:
                await handler(message)
            except Exception as exc:
                _log.error(
                    "bus.consume.handler_failed",
                    stream=key,
                    entry_id=msg_id_str,
                    error=str(exc),
                )
                await self._send_to_dlq(key, msg_id_str, fields, reason=f"handler: {exc}")
            await self._shard.xack(key, self._group, msg_id)
        _log.info(
            "bus.consume.ok",
            stream=key,
            entry_id=msg_id_str,
        )

    async def _send_to_dlq(
        self,
        original_stream: str,
        original_id: str,
        fields: dict[bytes, bytes],
        *,
        reason: str,
    ) -> None:
        await self._shard.xadd(
            self._shard.dlq_key(),
            {
                b"original_stream": original_stream.encode("utf-8"),
                b"original_id": original_id.encode("ascii"),
                b"reason": reason.encode("utf-8"),
                b"json": fields.get(b"json", b""),
            },
        )
