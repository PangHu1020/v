"""Unit tests for ``backend.app.wecom_aibot.outbound``."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest

from backend.app.wecom_aibot.outbound import WecomAibotOutbound


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


class TestWecomAibotOutbound:
    async def test_publishes_payload_on_pubsub(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        # Subscribe BEFORE publishing — pub/sub is fire-and-forget.
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("wecom_aibot:outbound")
        # Drop the subscribe-confirmation frame.
        await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)

        out = WecomAibotOutbound(redis_client, pubsub_channel="wecom_aibot:outbound")
        await out.send_text("ext-42", "hello world")

        msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)
        assert msg is not None
        payload = json.loads(msg["data"])
        assert payload == {"channel_user_id": "ext-42", "text": "hello world"}
        await pubsub.aclose()

    async def test_preserves_unicode_without_escaping(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("ch")
        await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)

        out = WecomAibotOutbound(redis_client, pubsub_channel="ch")
        await out.send_text("u", "你好世界")

        msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)
        # ensure_ascii=False means the CJK survives verbatim, not as \uXXXX.
        assert "你好世界".encode() in msg["data"]
        await pubsub.aclose()

    async def test_publish_with_no_subscribers_is_silent(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        out = WecomAibotOutbound(redis_client, pubsub_channel="nobody-listening")
        await out.send_text("u", "drop me")
