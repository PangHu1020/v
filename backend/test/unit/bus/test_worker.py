"""Unit tests for ``backend.app.bus.worker``.

Exercises the bus consumer handler against a real LangGraph (with mocked
LLMCaller), a real RedisCheckpointer over fakeredis, a stubbed PG pool,
and a captured ``send`` registry. Verifies session id resolution
(continuation under 30 min vs. fresh after silence) and outbound dispatch.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import fakeredis.aioredis
import pytest
from langchain_core.messages import AIMessage

from backend.app.bus.messages import SystemMessage
from backend.app.bus.worker import _resolve_session_id, make_bus_handler
from backend.v.agents.checkpointer import RedisCheckpointer
from backend.v.agents.graph import build_graph
from backend.v.models.llm_caller import LLMResult


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


def _fake_pool(profile_row: dict | None = None) -> MagicMock:
    pool = MagicMock(spec=asyncpg.Pool)
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=profile_row)

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    return pool


def _msg(text: str = "hello", user: str = "ext-1", channel: str = "wecom") -> SystemMessage:
    return SystemMessage(
        channel=channel,  # type: ignore[arg-type]
        channel_user_id=user,
        text=text,
        dedup_key=f"d-{user}-{text}",
    )


class TestSessionResolution:
    async def test_first_message_mints(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        sid, minted = await _resolve_session_id(
            redis_client,
            channel="wecom",
            channel_user_id="ext-fresh",
            silence_seconds=1800,
        )
        assert minted is True
        assert sid

    async def test_returning_within_window_reuses(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        sid1, _ = await _resolve_session_id(
            redis_client,
            channel="wecom",
            channel_user_id="ext-cont",
            silence_seconds=1800,
        )
        # Pretend the worker recorded last_seen.
        await redis_client.set("last_seen:wecom:ext-cont", str(time.time()), ex=3600)

        sid2, minted = await _resolve_session_id(
            redis_client,
            channel="wecom",
            channel_user_id="ext-cont",
            silence_seconds=1800,
        )
        assert minted is False
        assert sid2 == sid1

    async def test_returning_after_window_mints_new(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        sid1, _ = await _resolve_session_id(
            redis_client,
            channel="wecom",
            channel_user_id="ext-stale",
            silence_seconds=1800,
        )
        # Set last_seen far in the past.
        await redis_client.set("last_seen:wecom:ext-stale", str(time.time() - 4000), ex=3600)

        sid2, minted = await _resolve_session_id(
            redis_client,
            channel="wecom",
            channel_user_id="ext-stale",
            silence_seconds=1800,
        )
        assert minted is True
        assert sid2 != sid1


class TestHandlerEndToEnd:
    async def test_inbound_invokes_graph_and_dispatches_reply(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        ckpt = RedisCheckpointer(redis_client, ttl_seconds=600)
        graph = build_graph(ckpt)

        llm_caller = MagicMock()
        llm_caller.chat = AsyncMock(
            return_value=LLMResult(
                message=AIMessage(content="您好，可以帮您查询订单"),
                model="deepseek-flash",
                role="main_primary",
                fallback_used=False,
                latency_ms=15,
            )
        )

        sent: list[tuple[str, str]] = []

        async def wecom_send(channel_user_id: str, text: str) -> None:
            sent.append((channel_user_id, text))

        handler = make_bus_handler(
            graph=graph,
            pool=_fake_pool({"profile": {"member_level": "黄金"}}),
            redis=redis_client,
            llm_caller=llm_caller,
            sends={"wecom": wecom_send},
            silence_seconds=1800,
            cache_ttl_seconds=120,
        )

        await handler(_msg(text="订单 ORD123 状态？", user="ext-42"))

        assert sent == [("ext-42", "您好，可以帮您查询订单")]
        # llm_caller invoked once.
        assert llm_caller.chat.await_count == 1

    async def test_two_turns_within_session_share_thread_id(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        ckpt = RedisCheckpointer(redis_client, ttl_seconds=600)
        graph = build_graph(ckpt)

        llm_caller = MagicMock()
        llm_caller.chat = AsyncMock(
            return_value=LLMResult(
                message=AIMessage(content="ok"),
                model="m",
                role="main_primary",
                fallback_used=False,
                latency_ms=1,
            )
        )

        async def fake_send(_uid: str, _text: str) -> None:
            return None

        handler = make_bus_handler(
            graph=graph,
            pool=_fake_pool({"profile": {}}),
            redis=redis_client,
            llm_caller=llm_caller,
            sends={"wecom": fake_send},
            silence_seconds=1800,
            cache_ttl_seconds=120,
        )

        await handler(_msg(text="first", user="ext-cont"))
        await handler(_msg(text="second", user="ext-cont"))

        # Same session_id key value across both turns.
        sid_bytes = await redis_client.get("session:wecom:ext-cont")
        assert sid_bytes is not None
        # Two checkpoints under the same thread_id (same hash key).
        keys = await redis_client.keys(b"ckpt:*")
        assert len(keys) == 1

    async def test_unknown_channel_logs_and_does_not_send(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        ckpt = RedisCheckpointer(redis_client, ttl_seconds=600)
        graph = build_graph(ckpt)

        llm_caller = MagicMock()
        llm_caller.chat = AsyncMock(
            return_value=LLMResult(
                message=AIMessage(content="reply"),
                model="m",
                role="main_primary",
                fallback_used=False,
                latency_ms=1,
            )
        )

        handler = make_bus_handler(
            graph=graph,
            pool=_fake_pool({"profile": {}}),
            redis=redis_client,
            llm_caller=llm_caller,
            sends={},  # no adapters registered
            silence_seconds=1800,
            cache_ttl_seconds=120,
        )

        # Should not raise even without a registered send.
        await handler(_msg(text="hi"))
