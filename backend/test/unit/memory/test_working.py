"""Unit tests for ``backend.v.memory.working``."""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest

from backend.v.memory.types import MemoryEntry
from backend.v.memory.working import (
    append_working_memory,
    cache_user_profile,
    delete_working_memory,
    get_cached_user_profile,
    read_working_memory,
)


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


class TestProfileCache:
    async def test_set_then_get(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        profile = {"member_level": "黄金", "preferred_language": "zh"}
        await cache_user_profile(
            redis_client,
            channel="wecom",
            channel_user_id="ext-1",
            profile=profile,
            ttl_seconds=120,
        )
        retrieved = await get_cached_user_profile(
            redis_client, channel="wecom", channel_user_id="ext-1"
        )
        assert retrieved == profile

    async def test_get_returns_none_when_absent(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        retrieved = await get_cached_user_profile(
            redis_client, channel="feishu", channel_user_id="missing"
        )
        assert retrieved is None

    async def test_ttl_applied(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        await cache_user_profile(
            redis_client,
            channel="wecom",
            channel_user_id="ext-2",
            profile={"x": 1},
            ttl_seconds=42,
        )
        ttl = await redis_client.ttl("profile:wecom:ext-2")
        assert 0 < ttl <= 42

    async def test_chinese_round_trip(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        await cache_user_profile(
            redis_client,
            channel="wecom",
            channel_user_id="ext-zh",
            profile={"name": "张敏", "tier": "铂金"},
            ttl_seconds=60,
        )
        retrieved = await get_cached_user_profile(
            redis_client, channel="wecom", channel_user_id="ext-zh"
        )
        assert retrieved == {"name": "张敏", "tier": "铂金"}

    async def test_distinct_users_isolated(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        await cache_user_profile(
            redis_client,
            channel="wecom",
            channel_user_id="alice",
            profile={"v": 1},
            ttl_seconds=60,
        )
        await cache_user_profile(
            redis_client,
            channel="wecom",
            channel_user_id="bob",
            profile={"v": 2},
            ttl_seconds=60,
        )
        a = await get_cached_user_profile(redis_client, channel="wecom", channel_user_id="alice")
        b = await get_cached_user_profile(redis_client, channel="wecom", channel_user_id="bob")
        assert a == {"v": 1}
        assert b == {"v": 2}


class TestWorkingMemoryList:
    async def test_append_then_read_oldest_first(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        first = MemoryEntry(
            content="客户语气急切", importance=0.4, keywords=["情绪"], kind="observation"
        )
        second = MemoryEntry(
            content="客户偏好顺丰", importance=0.6, keywords=["快递"], kind="preference"
        )
        await append_working_memory(
            redis_client, session_id="s-1", entries=[first], ttl_seconds=120
        )
        await append_working_memory(
            redis_client, session_id="s-1", entries=[second], ttl_seconds=120
        )
        out = await read_working_memory(redis_client, session_id="s-1")
        assert [e.content for e in out] == [first.content, second.content]

    async def test_empty_entries_is_noop(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        await append_working_memory(redis_client, session_id="s-empty", entries=[], ttl_seconds=120)
        out = await read_working_memory(redis_client, session_id="s-empty")
        assert out == []

    async def test_read_missing_session_returns_empty(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        assert await read_working_memory(redis_client, session_id="ghost") == []

    async def test_distinct_sessions_isolated(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        a = MemoryEntry(content="A", importance=0.1, keywords=[], kind="observation")
        b = MemoryEntry(content="B", importance=0.1, keywords=[], kind="observation")
        await append_working_memory(redis_client, session_id="s-a", entries=[a], ttl_seconds=60)
        await append_working_memory(redis_client, session_id="s-b", entries=[b], ttl_seconds=60)
        assert (await read_working_memory(redis_client, session_id="s-a"))[0].content == "A"
        assert (await read_working_memory(redis_client, session_id="s-b"))[0].content == "B"

    async def test_delete_clears_list(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        e = MemoryEntry(content="x", importance=0.1, keywords=[], kind="observation")
        await append_working_memory(redis_client, session_id="s-d", entries=[e], ttl_seconds=60)
        await delete_working_memory(redis_client, session_id="s-d")
        assert await read_working_memory(redis_client, session_id="s-d") == []

    async def test_ttl_applied_on_append(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        e = MemoryEntry(content="x", importance=0.1, keywords=[], kind="observation")
        await append_working_memory(redis_client, session_id="s-ttl", entries=[e], ttl_seconds=42)
        ttl = await redis_client.ttl("working_memory:s-ttl")
        assert 0 < ttl <= 42

    async def test_corrupt_entry_skipped(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        await redis_client.rpush("working_memory:s-corrupt", b"not-json")
        e = MemoryEntry(content="ok", importance=0.1, keywords=[], kind="observation")
        await append_working_memory(
            redis_client, session_id="s-corrupt", entries=[e], ttl_seconds=60
        )
        out = await read_working_memory(redis_client, session_id="s-corrupt")
        assert [m.content for m in out] == ["ok"]
