"""Unit tests for ``backend.v.hooks.session``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import fakeredis.aioredis
import pytest

from backend.v.hooks.session import on_session_start


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


def _fake_pool(profile_row: dict | None) -> MagicMock:
    pool = MagicMock(spec=asyncpg.Pool)
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=profile_row)

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    return pool


class TestOnSessionStart:
    async def test_cache_hit_skips_pg(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        # Pre-warm the cache.
        from backend.v.memory.working import cache_user_profile

        await cache_user_profile(
            redis_client,
            channel="wecom",
            channel_user_id="ext-cached",
            profile={"member_level": "铂金"},
            ttl_seconds=120,
        )

        pool = _fake_pool(None)  # PG should NOT be hit.
        result = await on_session_start(
            pool=pool,
            redis=redis_client,
            channel="wecom",
            channel_user_id="ext-cached",
            cache_ttl_seconds=120,
        )
        assert result == {"member_level": "铂金"}

    async def test_cache_miss_warms_from_pg(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        pool = _fake_pool({"profile": {"customer_name": "李伟"}})
        result = await on_session_start(
            pool=pool,
            redis=redis_client,
            channel="wecom",
            channel_user_id="ext-fresh",
            cache_ttl_seconds=120,
        )
        assert result == {"customer_name": "李伟"}
        # Verify the cache was warmed.
        from backend.v.memory.working import get_cached_user_profile

        cached = await get_cached_user_profile(
            redis_client, channel="wecom", channel_user_id="ext-fresh"
        )
        assert cached == {"customer_name": "李伟"}

    async def test_cache_miss_unknown_user_returns_empty(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        pool = _fake_pool(None)
        result = await on_session_start(
            pool=pool,
            redis=redis_client,
            channel="feishu",
            channel_user_id="ou_new",
            cache_ttl_seconds=120,
        )
        assert result == {}

    async def test_distinct_users_isolated(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        pool = _fake_pool({"profile": {"who": "alice"}})
        a = await on_session_start(
            pool=pool,
            redis=redis_client,
            channel="wecom",
            channel_user_id="alice",
            cache_ttl_seconds=120,
        )
        # Now PG should still be hit for bob since cache is keyed per user.
        pool2 = _fake_pool({"profile": {"who": "bob"}})
        b = await on_session_start(
            pool=pool2,
            redis=redis_client,
            channel="wecom",
            channel_user_id="bob",
            cache_ttl_seconds=120,
        )
        assert a == {"who": "alice"}
        assert b == {"who": "bob"}
