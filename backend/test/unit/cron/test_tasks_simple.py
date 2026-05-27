"""Unit tests for ``backend.v.cron.tasks.logistics`` and ``ad_hoc_ad``."""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest

from backend.v.cron.tasks.ad_hoc_ad import push_ad
from backend.v.cron.tasks.logistics import notify_logistics_delivered


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


def _ctx_with_send(
    redis: fakeredis.aioredis.FakeRedis,
) -> tuple[dict, list[tuple[str, str]]]:
    sent: list[tuple[str, str]] = []

    async def fake_send(uid: str, text: str) -> None:
        sent.append((uid, text))

    return {"redis": redis, "sends": {"wecom": fake_send}}, sent


class TestNotifyLogisticsDelivered:
    async def test_message_includes_order_and_tracking(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        ctx, sent = _ctx_with_send(redis_client)
        ok = await notify_logistics_delivered(
            ctx,
            channel="wecom",
            channel_user_id="ext-1",
            order_id="ORD123",
            tracking_number="SF999",
        )
        assert ok is True
        assert len(sent) == 1
        _, text = sent[0]
        assert "ORD123" in text
        assert "SF999" in text
        assert "签收" in text

    async def test_courier_optional(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        ctx, sent = _ctx_with_send(redis_client)
        await notify_logistics_delivered(
            ctx,
            channel="wecom",
            channel_user_id="u",
            order_id="O",
            tracking_number="T",
            courier="顺丰",
        )
        _, text = sent[0]
        assert "顺丰" in text

        # Without courier, no spurious "承运商" line.
        ctx2, sent2 = _ctx_with_send(redis_client)
        await notify_logistics_delivered(
            ctx2,
            channel="wecom",
            channel_user_id="u2",
            order_id="O",
            tracking_number="T",
        )
        _, text2 = sent2[0]
        assert "承运商" not in text2

    async def test_unknown_channel_returns_false(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        ctx = {"redis": redis_client, "sends": {}}  # no wecom
        ok = await notify_logistics_delivered(
            ctx,
            channel="wecom",
            channel_user_id="u",
            order_id="O",
            tracking_number="T",
        )
        assert ok is False


class TestPushAd:
    async def test_forwards_text_unchanged(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        ctx, sent = _ctx_with_send(redis_client)
        ok = await push_ad(
            ctx,
            channel="wecom",
            channel_user_id="ext-ad",
            text="🎉 双十一专享 8 折，仅限今日。",
        )
        assert ok is True
        assert sent == [("ext-ad", "🎉 双十一专享 8 折，仅限今日。")]

    async def test_persists_to_proactive_log(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        ctx, _ = _ctx_with_send(redis_client)
        await push_ad(
            ctx,
            channel="wecom",
            channel_user_id="ext-log",
            text="hi",
        )
        items = await redis_client.lrange("proactive_log:wecom:ext-log", 0, -1)
        assert len(items) == 1
        assert b"ad_hoc_ad" in items[0]
