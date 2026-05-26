"""Sharded Redis Streams primitive.

Each ``(channel, channel_user_id)`` pair maps deterministically to one of
``shard_count`` Redis Streams via mmh3. Per-shard ordering plus a single
consumer per shard yields strict per-conversation serial processing while
distinct conversations parallelize across shards.
"""

from __future__ import annotations

from typing import Any

import mmh3
import redis.asyncio as redis_async


class RedisStreamShard:
    """Wrapper around Redis Streams that hashes shard assignment internally."""

    def __init__(
        self,
        client: redis_async.Redis,
        *,
        prefix: str,
        shard_count: int,
    ) -> None:
        if shard_count < 1:
            raise ValueError("shard_count must be >= 1")
        self._client = client
        self._prefix = prefix
        self._shard_count = shard_count

    @property
    def shard_count(self) -> int:
        return self._shard_count

    def shard_index(self, channel: str, channel_user_id: str) -> int:
        """Return the deterministic shard index for the given identity."""
        digest = mmh3.hash(f"{channel}:{channel_user_id}", signed=False)
        return digest % self._shard_count

    def stream_key(self, channel: str, channel_user_id: str) -> str:
        """Return the stream key holding messages for the given identity."""
        return f"{self._prefix}:{self.shard_index(channel, channel_user_id)}"

    def all_stream_keys(self) -> list[str]:
        """Return all shard stream keys, in shard-index order."""
        return [f"{self._prefix}:{i}" for i in range(self._shard_count)]

    def dlq_key(self) -> str:
        """Return the dead-letter queue stream key shared across shards."""
        return f"{self._prefix}:dlq"

    async def xadd(self, key: str, fields: dict[str, str | bytes]) -> bytes:
        """Append an entry to the given stream and return its id."""
        return await self._client.xadd(key, fields)

    async def ensure_group(self, key: str, group: str) -> None:
        """Idempotently create the consumer group at the stream tail."""
        try:
            await self._client.xgroup_create(key, group, id="0-0", mkstream=True)
        except redis_async.ResponseError as exc:
            # BUSYGROUP indicates the group already exists; any other error is fatal.
            if "BUSYGROUP" not in str(exc):
                raise

    async def xreadgroup(
        self,
        group: str,
        consumer: str,
        keys: list[str],
        *,
        count: int = 1,
        block_ms: int = 5000,
    ) -> list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]]:
        """Read the next pending entries for the consumer.

        Args:
            group: Consumer group name.
            consumer: Per-worker consumer identity (must be unique within the group).
            keys: Stream keys to read from. Each key uses the special ``>``
                ID so only never-delivered entries are returned.
            count: Maximum entries per stream.
            block_ms: How long to block when no entries are pending.

        Returns:
            A list of ``(stream_key, [(message_id, fields), ...])`` tuples.
        """
        streams = {k: ">" for k in keys}
        result: Any = await self._client.xreadgroup(
            group,
            consumer,
            streams,
            count=count,
            block=block_ms,
        )
        return list(result) if result else []

    async def xack(self, key: str, group: str, message_id: str | bytes) -> int:
        """Acknowledge a delivered message."""
        return await self._client.xack(key, group, message_id)
