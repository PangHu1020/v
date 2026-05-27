"""Unit tests for ``backend.v.cron.proactive``."""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest

from backend.v.cron.proactive import (
    PROACTIVE_LOG_KEY_PREFIX,
    deliver_proactive,
    record_proactive,
)


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


class TestRecordProactive:
    async def test_appends_to_per_identity_list(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        await record_proactive(
            redis_client,
            channel="wecom",
            channel_user_id="ext-1",
            text="您的订单已签收",
            purpose="logistics_delivered",
        )
        items = await redis_client.lrange(f"{PROACTIVE_LOG_KEY_PREFIX}:wecom:ext-1", 0, -1)
        assert len(items) == 1
        decoded = items[0].decode("utf-8")
        assert decoded.startswith("logistics_delivered\t")
        assert "您的订单已签收" in decoded

    async def test_distinct_users_isolated(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        await record_proactive(
            redis_client, channel="wecom", channel_user_id="alice", text="t1", purpose="p"
        )
        await record_proactive(
            redis_client, channel="wecom", channel_user_id="bob", text="t2", purpose="p"
        )
        a = await redis_client.lrange(f"{PROACTIVE_LOG_KEY_PREFIX}:wecom:alice", 0, -1)
        b = await redis_client.lrange(f"{PROACTIVE_LOG_KEY_PREFIX}:wecom:bob", 0, -1)
        assert len(a) == 1
        assert len(b) == 1
        assert b"t1" in a[0]
        assert b"t2" in b[0]

    async def test_ttl_applied(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        await record_proactive(
            redis_client,
            channel="wecom",
            channel_user_id="ttl-user",
            text="x",
            purpose="p",
        )
        ttl = await redis_client.ttl(f"{PROACTIVE_LOG_KEY_PREFIX}:wecom:ttl-user")
        # Default 30 days = 2_592_000 s.
        assert 0 < ttl <= 30 * 24 * 60 * 60


class TestDeliverProactive:
    async def test_records_and_dispatches(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        sent: list[tuple[str, str]] = []

        async def fake_send(uid: str, text: str) -> None:
            sent.append((uid, text))

        ok = await deliver_proactive(
            channel="wecom",
            channel_user_id="ext-2",
            text="hi",
            purpose="ad_hoc_ad",
            redis=redis_client,
            sends={"wecom": fake_send},
        )
        assert ok is True
        assert sent == [("ext-2", "hi")]

        items = await redis_client.lrange(f"{PROACTIVE_LOG_KEY_PREFIX}:wecom:ext-2", 0, -1)
        assert len(items) == 1
        assert b"ad_hoc_ad" in items[0]

    async def test_no_send_returns_false_and_skips_log(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        ok = await deliver_proactive(
            channel="discord",  # not registered
            channel_user_id="x",
            text="t",
            purpose="p",
            redis=redis_client,
            sends={"wecom": (lambda *_: None)},  # type: ignore[dict-item]
        )
        assert ok is False
        # Nothing logged for unrouted channel.
        items = await redis_client.lrange(f"{PROACTIVE_LOG_KEY_PREFIX}:discord:x", 0, -1)
        assert items == []
