"""Unit tests for ``backend.app.store.redis``.

Uses ``fakeredis`` so tests are hermetic and don't require a running Redis.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import fakeredis.aioredis
import pytest

from backend.app.store.redis import close_client, create_client, redis_health


@pytest.fixture
async def fake_redis() -> fakeredis.aioredis.FakeRedis:
    """Yield a fakeredis async client and aclose it after the test."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


class TestCreateClient:
    async def test_returns_async_redis_client(self) -> None:
        with patch("backend.app.store.redis.redis_async.from_url") as mock_from_url:
            mock_from_url.return_value = AsyncMock()
            client = await create_client("redis://localhost:6379/0")
            mock_from_url.assert_called_once_with(
                "redis://localhost:6379/0", decode_responses=False
            )
            assert client is mock_from_url.return_value


class TestRedisHealth:
    async def test_health_returns_true_when_ping_succeeds(
        self, fake_redis: fakeredis.aioredis.FakeRedis
    ) -> None:
        assert await redis_health(fake_redis) is True

    async def test_health_returns_false_when_ping_raises(self) -> None:
        client = AsyncMock()
        client.ping.side_effect = ConnectionError("down")
        assert await redis_health(client) is False


class TestRoundTrip:
    async def test_set_and_get(self, fake_redis: fakeredis.aioredis.FakeRedis) -> None:
        await fake_redis.set("k", b"v")
        assert await fake_redis.get("k") == b"v"

    async def test_close_client_does_not_raise(
        self, fake_redis: fakeredis.aioredis.FakeRedis
    ) -> None:
        # close_client uses the same aclose() under the hood; calling it twice
        # on fakeredis is safe.
        await close_client(fake_redis)
