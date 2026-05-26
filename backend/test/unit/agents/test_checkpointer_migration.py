"""Unit tests for ``backend.v.agents.checkpointer_migration``.

Uses two ``RedisCheckpointer`` instances backed by separate fakeredis
clients (one acts as "hot", the other as "cold"). The migration helpers
only depend on the BaseCheckpointSaver interface so this exercise the
logic without needing the real Postgres saver.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from backend.v.agents.checkpointer import RedisCheckpointer
from backend.v.agents.checkpointer_migration import migrate_cold_to_hot, migrate_hot_to_cold


@pytest.fixture
async def hot_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


@pytest.fixture
async def cold_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


@pytest.fixture
def hot(hot_redis: fakeredis.aioredis.FakeRedis) -> RedisCheckpointer:
    return RedisCheckpointer(hot_redis, ttl_seconds=600)


@pytest.fixture
def cold(cold_redis: fakeredis.aioredis.FakeRedis) -> RedisCheckpointer:
    return RedisCheckpointer(cold_redis, ttl_seconds=86400)


def _config(thread_id: str, parent: str | None = None) -> RunnableConfig:
    cfg: dict = {"thread_id": thread_id, "checkpoint_ns": ""}
    if parent:
        cfg["checkpoint_id"] = parent
    return {"configurable": cfg}


def _checkpoint(cid: str, value: str) -> Checkpoint:
    return {
        "v": 4,
        "id": cid,
        "ts": "2026-05-26T00:00:00+00:00",
        "channel_values": {"foo": value},
        "channel_versions": {"foo": "1"},
        "versions_seen": {},
        "pending_sends": [],
    }


async def _seed_three_checkpoints(ckpt: RedisCheckpointer, thread_id: str) -> list[str]:
    ids = ["c-0", "c-1", "c-2"]
    parents = [None, "c-0", "c-1"]
    for cid, parent in zip(ids, parents, strict=False):
        meta: CheckpointMetadata = {"step": int(cid.split("-")[1])}
        await ckpt.aput(_config(thread_id, parent), _checkpoint(cid, cid), meta, {})
    return ids


class TestMigrateHotToCold:
    async def test_replays_oldest_first_and_drops_source(
        self,
        hot: RedisCheckpointer,
        cold: RedisCheckpointer,
        hot_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        await _seed_three_checkpoints(hot, "thread-X")
        count = await migrate_hot_to_cold(
            "thread-X",
            redis_ckpt=hot,
            pg_ckpt=cold,
        )
        assert count == 3

        # Latest checkpoint reachable in cold.
        cold_latest = await cold.aget_tuple(_config("thread-X"))
        assert cold_latest is not None
        assert cold_latest.checkpoint["id"] == "c-2"
        # Parent chain preserved.
        assert cold_latest.parent_config is not None
        assert cold_latest.parent_config["configurable"]["checkpoint_id"] == "c-1"

        # Hot is empty.
        assert await hot.aget_tuple(_config("thread-X")) is None
        assert await hot_redis.exists("ckpt:thread-X") == 0

    async def test_other_threads_untouched(
        self,
        hot: RedisCheckpointer,
        cold: RedisCheckpointer,
    ) -> None:
        await _seed_three_checkpoints(hot, "thread-A")
        await _seed_three_checkpoints(hot, "thread-B")

        await migrate_hot_to_cold("thread-A", redis_ckpt=hot, pg_ckpt=cold)

        assert await hot.aget_tuple(_config("thread-A")) is None
        assert await hot.aget_tuple(_config("thread-B")) is not None
        assert await cold.aget_tuple(_config("thread-A")) is not None
        assert await cold.aget_tuple(_config("thread-B")) is None


class TestMigrateColdToHot:
    async def test_round_trip(
        self,
        hot: RedisCheckpointer,
        cold: RedisCheckpointer,
    ) -> None:
        await _seed_three_checkpoints(hot, "thread-rt")
        await migrate_hot_to_cold("thread-rt", redis_ckpt=hot, pg_ckpt=cold)
        # Now cold has 3, hot has 0.
        await migrate_cold_to_hot("thread-rt", pg_ckpt=cold, redis_ckpt=hot)

        # Hot reachable again.
        latest = await hot.aget_tuple(_config("thread-rt"))
        assert latest is not None
        assert latest.checkpoint["id"] == "c-2"
        # Cold drained.
        assert await cold.aget_tuple(_config("thread-rt")) is None

    async def test_missing_thread_is_zero(
        self,
        hot: RedisCheckpointer,
        cold: RedisCheckpointer,
    ) -> None:
        # Migrating a thread that exists in neither store is a no-op zero.
        count = await migrate_hot_to_cold("ghost", redis_ckpt=hot, pg_ckpt=cold)
        assert count == 0


class TestIdempotency:
    async def test_double_hot_to_cold_does_not_error(
        self,
        hot: RedisCheckpointer,
        cold: RedisCheckpointer,
    ) -> None:
        await _seed_three_checkpoints(hot, "thread-idem")
        await migrate_hot_to_cold("thread-idem", redis_ckpt=hot, pg_ckpt=cold)
        # Running again finds an empty source and is a no-op.
        count = await migrate_hot_to_cold("thread-idem", redis_ckpt=hot, pg_ckpt=cold)
        assert count == 0

    async def test_partial_failure_leaves_thread_reachable(
        self,
        hot: RedisCheckpointer,
        cold: RedisCheckpointer,
    ) -> None:
        # Simulate that migration succeeded in copying to cold but crashed
        # before deleting hot. Both stores should still resolve the thread.
        await _seed_three_checkpoints(hot, "thread-partial")
        # Manually replay without delete (mimics partial failure).
        config = _config("thread-partial")
        tuples = []
        async for t in hot.alist(config):
            tuples.append(t)
        for t in reversed(tuples):
            await cold.aput(t.config, t.checkpoint, t.metadata, {})
        # NOT deleting from hot.

        # Both stores resolve to the same latest id.
        hot_latest = await hot.aget_tuple(config)
        cold_latest = await cold.aget_tuple(config)
        assert hot_latest is not None and cold_latest is not None
        assert hot_latest.checkpoint["id"] == cold_latest.checkpoint["id"]

        # A retry of hot_to_cold cleans up.
        await migrate_hot_to_cold("thread-partial", redis_ckpt=hot, pg_ckpt=cold)
        assert await hot.aget_tuple(config) is None
