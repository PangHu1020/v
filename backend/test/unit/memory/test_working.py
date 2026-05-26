"""Unit tests for ``backend.v.memory.working``."""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest

from backend.v.memory.working import cache_user_profile, get_cached_user_profile


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
