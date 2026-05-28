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


def _fake_pool(
    profile_row: dict | None = None,
    event_rows: list[dict] | None = None,
) -> MagicMock:
    pool = MagicMock(spec=asyncpg.Pool)
    conn = MagicMock()

    async def _fetchrow(sql: str, *args):
        if "FROM agent.user_profile" in sql:
            return profile_row
        return None

    async def _fetch(sql: str, *args):
        if "FROM agent.session_memory" in sql:
            return event_rows or []
        return []

    conn.fetchrow = AsyncMock(side_effect=_fetchrow)
    conn.fetch = AsyncMock(side_effect=_fetch)

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    return pool


class TestOnSessionStartProfile:
    async def test_cache_hit_skips_pg(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        from backend.v.memory.working import cache_user_profile

        await cache_user_profile(
            redis_client,
            channel="wecom",
            channel_user_id="ext-cached",
            profile={"member_level": "铂金"},
            ttl_seconds=120,
        )

        pool = _fake_pool(None)
        result = await on_session_start(
            pool=pool,
            redis=redis_client,
            channel="wecom",
            channel_user_id="ext-cached",
            cache_ttl_seconds=120,
        )
        assert result["profile"] == {"member_level": "铂金"}
        assert result["recent_events"] == []  # not requested

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
        assert result["profile"] == {"customer_name": "李伟"}
        # Cache was warmed.
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
        assert result["profile"] == {}
        assert result["recent_events"] == []

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
        pool2 = _fake_pool({"profile": {"who": "bob"}})
        b = await on_session_start(
            pool=pool2,
            redis=redis_client,
            channel="wecom",
            channel_user_id="bob",
            cache_ttl_seconds=120,
        )
        assert a["profile"] == {"who": "alice"}
        assert b["profile"] == {"who": "bob"}


class TestOnSessionStartRecentEvents:
    async def test_zero_limit_skips_query(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        from datetime import datetime

        # Even if rows exist, ``recent_events_limit=0`` returns [] without
        # querying.
        pool = _fake_pool(
            {"profile": {}},
            event_rows=[
                {
                    "summary": "上次咨询了订单",
                    "metadata": {},
                    "created_at": datetime.now(),
                }
            ],
        )
        result = await on_session_start(
            pool=pool,
            redis=redis_client,
            channel="wecom",
            channel_user_id="u",
            cache_ttl_seconds=120,
            recent_events_limit=0,
        )
        assert result["recent_events"] == []

    async def test_returns_recent_events(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        from datetime import datetime, timedelta

        rows = [
            {
                "summary": f"event-{i}",
                "metadata": {"intents": ["x"]},
                "created_at": datetime.now() - timedelta(days=i),
            }
            for i in range(3)
        ]
        pool = _fake_pool({"profile": {}}, event_rows=rows)
        result = await on_session_start(
            pool=pool,
            redis=redis_client,
            channel="wecom",
            channel_user_id="u",
            cache_ttl_seconds=120,
            recent_events_limit=3,
        )
        assert len(result["recent_events"]) == 3
        assert result["recent_events"][0]["summary"] == "event-0"

    async def test_no_events_returns_empty_list(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        pool = _fake_pool({"profile": {}}, event_rows=[])
        result = await on_session_start(
            pool=pool,
            redis=redis_client,
            channel="wecom",
            channel_user_id="u",
            cache_ttl_seconds=120,
            recent_events_limit=5,
        )
        assert result["recent_events"] == []
