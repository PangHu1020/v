"""Unit tests for ``backend.app.operator.slack.outbound``."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import pytest
from slack_sdk.errors import SlackApiError

from backend.app.operator.slack.outbound import SlackOutbound, SlackOutboundError


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


def _ok_post_message(ts: str = "1700000000.000100") -> dict:
    return {"ok": True, "ts": ts, "channel": "C123"}


def _make_client_mock(post_response: dict | Exception) -> MagicMock:
    client = MagicMock()
    if isinstance(post_response, Exception):
        client.chat_postMessage = AsyncMock(side_effect=post_response)
    else:
        client.chat_postMessage = AsyncMock(return_value=post_response)
    client.close = AsyncMock()
    return client


class TestPostHandoffAlert:
    async def test_records_thread_mapping_in_redis(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        client = _make_client_mock(_ok_post_message("1700.001"))
        out = SlackOutbound(
            bot_token="xoxb-x",
            alert_channel_id="C123",
            redis=redis_client,
            client=client,
        )

        ts = await out.post_handoff_alert(
            session_id="sess-42",
            channel="wecom",
            channel_user_id="ext-1",
            customer_text="退货怎么操作？",
            transfer_reason="超出 SOP",
        )
        assert ts == "1700.001"

        # Forward mapping: thread_ts -> session_id.
        forward = await redis_client.get("slack_thread:1700.001")
        assert forward.decode("utf-8") == "sess-42"
        # Reverse mapping: session_id -> {channel_id, thread_ts}.
        reverse = await redis_client.get("session_thread:sess-42")
        data = json.loads(reverse)
        assert data == {"channel_id": "C123", "thread_ts": "1700.001"}

    async def test_blocks_include_session_in_button_value(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        client = _make_client_mock(_ok_post_message())
        out = SlackOutbound(
            bot_token="xoxb-x",
            alert_channel_id="C",
            redis=redis_client,
            client=client,
        )
        await out.post_handoff_alert(
            session_id="sess-button",
            channel="wecom",
            channel_user_id="ext",
            customer_text="hi",
            transfer_reason="r",
        )
        kwargs = client.chat_postMessage.call_args.kwargs
        blocks = kwargs["blocks"]
        # Find the action block's button value.
        actions = next(b for b in blocks if b.get("type") == "actions")
        button = actions["elements"][0]
        assert button["action_id"] == "resume_session"
        assert button["value"] == "sess-button"

    async def test_slack_failure_raises(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        err = SlackApiError("slack down", response=MagicMock(data={"error": "rate_limited"}))
        client = _make_client_mock(err)
        out = SlackOutbound(
            bot_token="xoxb-x",
            alert_channel_id="C",
            redis=redis_client,
            client=client,
        )
        with pytest.raises(SlackOutboundError, match=r"chat\.postMessage failed"):
            await out.post_handoff_alert(
                session_id="x",
                channel="wecom",
                channel_user_id="u",
                customer_text="t",
                transfer_reason="r",
            )


class TestPostCustomerMessage:
    async def test_posts_to_thread(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        client = _make_client_mock(_ok_post_message())
        out = SlackOutbound(
            bot_token="xoxb-x",
            alert_channel_id="Cabc",
            redis=redis_client,
            client=client,
        )
        await out.post_customer_message(thread_ts="1700.5", text="客户继续问")
        kwargs = client.chat_postMessage.call_args.kwargs
        assert kwargs["channel"] == "Cabc"
        assert kwargs["thread_ts"] == "1700.5"
        assert "客户继续问" in kwargs["text"]


class TestGetThreadForSession:
    async def test_returns_thread_when_present(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        client = _make_client_mock(_ok_post_message("1700.7"))
        out = SlackOutbound(
            bot_token="xoxb-x",
            alert_channel_id="C",
            redis=redis_client,
            client=client,
        )
        await out.post_handoff_alert(
            session_id="s7",
            channel="wecom",
            channel_user_id="u",
            customer_text="t",
            transfer_reason="r",
        )
        ts = await out.get_thread_for_session("s7")
        assert ts == "1700.7"

    async def test_returns_none_when_unknown(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        out = SlackOutbound(
            bot_token="xoxb-x",
            alert_channel_id="C",
            redis=redis_client,
            client=_make_client_mock(_ok_post_message()),
        )
        ts = await out.get_thread_for_session("nope")
        assert ts is None
