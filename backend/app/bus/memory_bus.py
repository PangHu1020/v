"""Durable memory-task streams: extraction, promotion, and consolidation.

Replaces the two ``asyncio.create_task`` fire-and-forget calls in the bus
worker with at-least-once Redis Streams delivery so background memory jobs
survive process restarts.

Three streams (single global, not sharded — throughput is low and ordering
within a session is ensured by idempotent handlers):

    memory:promote     session_id to promote (session-end → durable stores)
    memory:consolidate channel + user_id to consolidate (monthly forgetting)
    memory:extract     session idle → pre-populate Redis working memory

Idle detection: after each agent turn, ``mark_turn_active`` records the
session in a Redis sorted set (score = unix timestamp). ``IdleWatcher`` runs
every ``interval`` seconds, finds sessions silent for ``threshold`` seconds,
and enqueues an extraction task.
"""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import Awaitable, Callable
from typing import Any

import redis.asyncio as redis_async

from backend.v.utils.logging import get_logger

_log = get_logger("bus.memory")

PROMOTE_STREAM = "memory:promote"
CONSOLIDATE_STREAM = "memory:consolidate"
EXTRACT_STREAM = "memory:extract"
IDLE_ZSET = "memory:idle_watch"
DLQ_STREAM = "memory:dlq"
GROUP = "memory_workers"

_ALL_STREAMS = (PROMOTE_STREAM, CONSOLIDATE_STREAM, EXTRACT_STREAM)

IDLE_THRESHOLD_SECONDS = 300  # 5 min of silence before extraction


# ── Enqueue helpers (called by bus worker after/before each turn) ─────────────


async def enqueue_promote(redis: redis_async.Redis, *, session_id: str) -> None:
    """Enqueue session-end promotion (replaces fire-and-forget create_task)."""
    await redis.xadd(PROMOTE_STREAM, {"session_id": session_id})


async def enqueue_consolidate(redis: redis_async.Redis, *, channel: str, user_id: str) -> None:
    """Enqueue monthly episodic consolidation for a returning customer."""
    await redis.xadd(CONSOLIDATE_STREAM, {"channel": channel, "user_id": user_id})


async def mark_turn_active(
    redis: redis_async.Redis, *, session_id: str, channel: str, user_id: str
) -> None:
    """Record the session as just-active; ZADD overwrites the timestamp on each turn."""
    member = f"{session_id}:{channel}:{user_id}"
    await redis.zadd(IDLE_ZSET, {member: time.time()})


# ── Consumer ─────────────────────────────────────────────────────────────────

# Handler type: receives the decoded fields dict, returns None.
MemTaskHandler = Callable[[dict[str, str]], Awaitable[None]]


class MemoryConsumer:
    """Single asyncio task consuming all three memory-task streams.

    Reads one entry at a time from each stream (``count=1``), dispatches to
    the appropriate handler, then XACKs. Failures route to the DLQ stream and
    are still ACKed so the consumer group does not redeliver indefinitely.
    """

    def __init__(
        self,
        redis: redis_async.Redis,
        *,
        promote_handler: MemTaskHandler,
        consolidate_handler: MemTaskHandler,
        extract_handler: MemTaskHandler,
        block_ms: int = 5000,
    ) -> None:
        self._redis = redis
        self._handlers: dict[str, MemTaskHandler] = {
            PROMOTE_STREAM: promote_handler,
            CONSOLIDATE_STREAM: consolidate_handler,
            EXTRACT_STREAM: extract_handler,
        }
        self._block_ms = block_ms
        self._stop = asyncio.Event()
        self._consumer_name = f"{socket.gethostname()}:{id(self)}"

    async def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        for stream in _ALL_STREAMS:
            try:
                await self._redis.xgroup_create(stream, GROUP, id="0", mkstream=True)
            except Exception:  # noqa: S110 — BUSYGROUP means group already exists, safe to ignore
                pass

        streams = {s: ">" for s in _ALL_STREAMS}
        while not self._stop.is_set():
            entries = await self._redis.xreadgroup(
                GROUP,
                self._consumer_name,
                streams,
                count=1,
                block=self._block_ms,
            )
            if not entries:
                await asyncio.sleep(0.01)
                continue
            for raw_stream, batch in entries:
                stream_key = raw_stream.decode() if isinstance(raw_stream, bytes) else raw_stream
                handler = self._handlers.get(stream_key)
                for msg_id, raw_fields in batch:
                    fields = {
                        (k.decode() if isinstance(k, bytes) else k): (
                            v.decode() if isinstance(v, bytes) else v
                        )
                        for k, v in raw_fields.items()
                    }
                    await self._dispatch(stream_key, msg_id, fields, handler)

    async def _dispatch(
        self,
        stream_key: str,
        msg_id: Any,
        fields: dict[str, str],
        handler: MemTaskHandler | None,
    ) -> None:
        try:
            if handler:
                await handler(fields)
        except Exception as exc:
            _log.error(
                "memory_bus.handler_failed",
                stream=stream_key,
                error=type(exc).__name__,
                fields=fields,
            )
            await self._redis.xadd(
                DLQ_STREAM,
                {
                    "stream": stream_key,
                    "msg_id": str(msg_id),
                    "error": str(exc),
                    **fields,
                },
            )
        finally:
            await self._redis.xack(stream_key, GROUP, msg_id)


# ── Idle watcher ─────────────────────────────────────────────────────────────


class IdleWatcher:
    """Background loop: detect sessions idle for ``threshold`` seconds and
    enqueue an extraction task so working memory is populated before session-end.
    """

    def __init__(
        self,
        redis: redis_async.Redis,
        *,
        interval: int = 60,
        threshold: int = IDLE_THRESHOLD_SECONDS,
    ) -> None:
        self._redis = redis
        self._interval = interval
        self._threshold = threshold
        self._stop = asyncio.Event()

    async def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                cutoff = time.time() - self._threshold
                members: list[Any] = await self._redis.zrangebyscore(IDLE_ZSET, 0, cutoff)
                if members:
                    for m in members:
                        raw = m.decode() if isinstance(m, bytes) else m
                        parts = raw.split(":", 2)
                        if len(parts) == 3:
                            session_id, channel, user_id = parts
                            await self._redis.xadd(
                                EXTRACT_STREAM,
                                {
                                    "session_id": session_id,
                                    "channel": channel,
                                    "user_id": user_id,
                                },
                            )
                    await self._redis.zremrangebyscore(IDLE_ZSET, 0, cutoff)
                    _log.info("memory_bus.idle_enqueued", count=len(members))
            except Exception as exc:
                _log.error("memory_bus.idle_watcher_error", error=type(exc).__name__)

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except TimeoutError:
                pass
