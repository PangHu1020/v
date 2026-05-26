"""Unit tests for ``backend.v.memory.checkpointer``."""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from backend.v.memory.checkpointer import RedisCheckpointer


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


@pytest.fixture
def checkpointer(redis_client: fakeredis.aioredis.FakeRedis) -> RedisCheckpointer:
    return RedisCheckpointer(redis_client, ttl_seconds=1800)


def _config(thread_id: str, checkpoint_id: str | None = None) -> RunnableConfig:
    cfg: dict = {"thread_id": thread_id, "checkpoint_ns": ""}
    if checkpoint_id:
        cfg["checkpoint_id"] = checkpoint_id
    return {"configurable": cfg}


def _checkpoint(checkpoint_id: str, value: str = "value") -> Checkpoint:
    return {
        "v": 4,
        "id": checkpoint_id,
        "ts": "2026-05-26T00:00:00+00:00",
        "channel_values": {"foo": value},
        "channel_versions": {"foo": "1"},
        "versions_seen": {},
        "pending_sends": [],
    }


class TestRoundTrip:
    async def test_put_then_get_latest(self, checkpointer: RedisCheckpointer) -> None:
        cfg = _config("t-1")
        ckpt = _checkpoint("c-1", value="first")
        meta: CheckpointMetadata = {"source": "input", "step": 0, "writes": {}}
        new_cfg = await checkpointer.aput(cfg, ckpt, meta, {})
        assert new_cfg["configurable"]["checkpoint_id"] == "c-1"

        retrieved = await checkpointer.aget_tuple(_config("t-1"))
        assert retrieved is not None
        assert retrieved.checkpoint["id"] == "c-1"
        assert retrieved.checkpoint["channel_values"]["foo"] == "first"
        assert retrieved.metadata.get("source") == "input"

    async def test_put_overwrites_latest_pointer(self, checkpointer: RedisCheckpointer) -> None:
        cfg = _config("t-2")
        await checkpointer.aput(cfg, _checkpoint("c-1", "v1"), {"step": 0}, {})
        await checkpointer.aput(
            _config("t-2", "c-1"),
            _checkpoint("c-2", "v2"),
            {"step": 1},
            {},
        )
        latest = await checkpointer.aget_tuple(_config("t-2"))
        assert latest is not None
        assert latest.checkpoint["id"] == "c-2"
        assert latest.parent_config is not None
        assert latest.parent_config["configurable"]["checkpoint_id"] == "c-1"

    async def test_get_specific_checkpoint(self, checkpointer: RedisCheckpointer) -> None:
        cfg = _config("t-3")
        await checkpointer.aput(cfg, _checkpoint("c-1", "v1"), {"step": 0}, {})
        await checkpointer.aput(_config("t-3", "c-1"), _checkpoint("c-2", "v2"), {"step": 1}, {})
        old = await checkpointer.aget_tuple(_config("t-3", "c-1"))
        assert old is not None
        assert old.checkpoint["channel_values"]["foo"] == "v1"


class TestList:
    async def test_alist_returns_newest_first(self, checkpointer: RedisCheckpointer) -> None:
        cfg = _config("t-4")
        for i in range(3):
            await checkpointer.aput(
                _config("t-4", f"c-{i - 1}" if i else None),
                _checkpoint(f"c-{i}", str(i)),
                {"step": i},
                {},
            )
        items = []
        async for tup in checkpointer.alist(cfg, limit=10):
            items.append(tup.checkpoint["id"])
        assert items == ["c-2", "c-1", "c-0"]

    async def test_alist_respects_limit(self, checkpointer: RedisCheckpointer) -> None:
        cfg = _config("t-5")
        for i in range(5):
            await checkpointer.aput(_config("t-5"), _checkpoint(f"c-{i}"), {"step": i}, {})
        items = []
        async for tup in checkpointer.alist(cfg, limit=2):
            items.append(tup.checkpoint["id"])
        assert len(items) == 2


class TestTtl:
    async def test_ttl_set_on_thread_hash(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
        checkpointer: RedisCheckpointer,
    ) -> None:
        await checkpointer.aput(_config("t-6"), _checkpoint("c-1"), {"step": 0}, {})
        ttl = await redis_client.ttl("ckpt:t-6")
        assert 0 < ttl <= 1800


class TestDeleteThread:
    async def test_removes_all_thread_state(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
        checkpointer: RedisCheckpointer,
    ) -> None:
        await checkpointer.aput(_config("t-7"), _checkpoint("c-1"), {"step": 0}, {})
        await checkpointer.aput_writes(
            _config("t-7", "c-1"), [("messages", {"k": "v"})], task_id="task-1"
        )
        # Confirm the key exists, delete the thread, confirm it's gone.
        assert await redis_client.exists("ckpt:t-7")
        await checkpointer.adelete_thread("t-7")
        assert await redis_client.exists("ckpt:t-7") == 0


class TestPutWrites:
    async def test_writes_persisted_separately(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
        checkpointer: RedisCheckpointer,
    ) -> None:
        await checkpointer.aput(_config("t-8"), _checkpoint("c-1"), {"step": 0}, {})
        await checkpointer.aput_writes(
            _config("t-8", "c-1"),
            [("messages", {"role": "user"}), ("scratch", "anything")],
            task_id="task-9",
        )
        keys = []
        async for k in redis_client.scan_iter(match="ckpt_writes:t-8:*"):
            keys.append(k)
        assert keys


class TestMissingThread:
    async def test_returns_none(self, checkpointer: RedisCheckpointer) -> None:
        assert await checkpointer.aget_tuple(_config("nope")) is None
