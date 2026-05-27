"""Unit tests for ``backend.v.mcp.cache``."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import fakeredis.aioredis
import orjson
import pytest

from backend.v.mcp.cache import MCPToolCache, _cache_key


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


@pytest.fixture
def cache(redis_client: fakeredis.aioredis.FakeRedis) -> MCPToolCache:
    return MCPToolCache(redis_client, l1_ttl_seconds=60, l2_ttl_seconds=3600)


class TestCacheKey:
    def test_canonicalizes_argument_order(self) -> None:
        a = _cache_key("server", "tool", {"a": 1, "b": 2})
        b = _cache_key("server", "tool", {"b": 2, "a": 1})
        assert a == b

    def test_distinguishes_servers(self) -> None:
        a = _cache_key("server-a", "tool", {"x": 1})
        b = _cache_key("server-b", "tool", {"x": 1})
        assert a != b

    def test_distinguishes_tools(self) -> None:
        a = _cache_key("server", "tool-a", {"x": 1})
        b = _cache_key("server", "tool-b", {"x": 1})
        assert a != b

    def test_distinguishes_args(self) -> None:
        a = _cache_key("server", "tool", {"x": 1})
        b = _cache_key("server", "tool", {"x": 2})
        assert a != b

    def test_chinese_args(self) -> None:
        # Should not crash and should round-trip stably.
        k1 = _cache_key("s", "t", {"name": "李伟"})
        k2 = _cache_key("s", "t", {"name": "李伟"})
        assert k1 == k2


class TestPutGet:
    async def test_round_trip(self, cache: MCPToolCache) -> None:
        await cache.put(
            server_id="s", tool_name="t", arguments={"id": "P001"}, value={"name": "iPhone"}
        )
        result = await cache.get(server_id="s", tool_name="t", arguments={"id": "P001"})
        assert result == {"name": "iPhone"}

    async def test_miss_returns_none(self, cache: MCPToolCache) -> None:
        result = await cache.get(server_id="s", tool_name="t", arguments={"id": "P404"})
        assert result is None

    async def test_l1_serves_after_l2_cleared(
        self,
        cache: MCPToolCache,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        await cache.put(server_id="s", tool_name="t", arguments={"x": 1}, value="v")
        # Wipe L2; L1 must still answer.
        await redis_client.flushdb()
        result = await cache.get(server_id="s", tool_name="t", arguments={"x": 1})
        assert result == "v"

    async def test_l1_miss_falls_back_to_l2(
        self,
        cache: MCPToolCache,
    ) -> None:
        await cache.put(server_id="s", tool_name="t", arguments={"x": 1}, value="v")
        cache.clear_l1()
        result = await cache.get(server_id="s", tool_name="t", arguments={"x": 1})
        assert result == "v"

    async def test_l2_hit_warms_l1(
        self,
        cache: MCPToolCache,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        # Populate L2 directly, bypassing L1.
        key = _cache_key("s", "t", {"x": 1})
        await redis_client.set(key, orjson.dumps("from-l2"), ex=3600)
        # First get: L1 miss, L2 hit.
        first = await cache.get(server_id="s", tool_name="t", arguments={"x": 1})
        assert first == "from-l2"
        # Second get: L1 hit; wipe L2 to confirm.
        await redis_client.flushdb()
        second = await cache.get(server_id="s", tool_name="t", arguments={"x": 1})
        assert second == "from-l2"


class TestL1Expiry:
    async def test_l1_expires(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        cache = MCPToolCache(redis_client, l1_ttl_seconds=0, l2_ttl_seconds=3600)
        await cache.put(server_id="s", tool_name="t", arguments={"x": 1}, value="v")
        # Force L1 expiry by waiting one monotonic tick.
        time.sleep(0.001)
        # L1 entry should be gone; L2 still has it.
        assert cache._l1_get(_cache_key("s", "t", {"x": 1})) is None
        result = await cache.get(server_id="s", tool_name="t", arguments={"x": 1})
        assert result == "v"


class TestInvalidate:
    async def test_drops_both_tiers(self, cache: MCPToolCache) -> None:
        await cache.put(server_id="s", tool_name="t", arguments={"x": 1}, value="v")
        await cache.invalidate(server_id="s", tool_name="t", arguments={"x": 1})
        assert (await cache.get(server_id="s", tool_name="t", arguments={"x": 1})) is None


class TestL2TtlApplied:
    async def test_redis_key_has_ttl(
        self,
        cache: MCPToolCache,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        await cache.put(server_id="s", tool_name="t", arguments={"x": 1}, value="v")
        key = _cache_key("s", "t", {"x": 1})
        ttl = await redis_client.ttl(key)
        assert 0 < ttl <= 3600
