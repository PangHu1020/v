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
from backend.v.agents.checkpoints.redis import RedisCheckpointer
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
    conn.execute = AsyncMock(return_value="OK")

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
        # llm_caller invoked at least once (intent + agent + reflection all call it).
        assert llm_caller.chat.await_count >= 1

    async def test_multi_segment_reply_splits_on_blank_line(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        """A reply with ``\\n\\n`` splits into multiple outbound sends."""
        ckpt = RedisCheckpointer(redis_client, ttl_seconds=600)
        graph = build_graph(ckpt)

        llm_caller = MagicMock()
        llm_caller.chat = AsyncMock(
            return_value=LLMResult(
                message=AIMessage(
                    content="第一步：登录账户\n\n第二步：进入订单页\n\n第三步：点击退款"
                ),
                model="m",
                role="main_primary",
                fallback_used=False,
                latency_ms=1,
            )
        )

        sent: list[tuple[str, str]] = []

        async def wecom_send(channel_user_id: str, text: str) -> None:
            sent.append((channel_user_id, text))

        handler = make_bus_handler(
            graph=graph,
            pool=_fake_pool({"profile": {}}),
            redis=redis_client,
            llm_caller=llm_caller,
            sends={"wecom": wecom_send},
            silence_seconds=1800,
            cache_ttl_seconds=120,
        )

        await handler(_msg(text="如何退款？", user="ext-99"))

        assert sent == [
            ("ext-99", "第一步：登录账户"),
            ("ext-99", "第二步：进入订单页"),
            ("ext-99", "第三步：点击退款"),
        ]

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


class TestSuspendedSessionRouting:
    """Phase-2 P0.D: when a session is suspended, customer messages bypass
    the graph and forward to the Slack alert thread instead."""

    async def test_forwards_to_slack_thread_when_suspended(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        ckpt = RedisCheckpointer(redis_client, ttl_seconds=600)
        graph = build_graph(ckpt)

        llm_caller = MagicMock()
        llm_caller.chat = AsyncMock()  # MUST NOT be called

        forwarded: list[tuple[str, str]] = []

        class _StubSlack:
            async def get_thread_for_session(self, session_id):
                return "1700.5"

            async def post_customer_message(self, *, thread_ts, text):
                forwarded.append((thread_ts, text))

            async def post_handoff_alert(self, **kwargs):
                return "ts"

        slack = _StubSlack()

        handler = make_bus_handler(
            graph=graph,
            pool=_fake_pool({"profile": {}}),
            redis=redis_client,
            llm_caller=llm_caller,
            sends={"wecom": AsyncMock()},
            silence_seconds=1800,
            cache_ttl_seconds=120,
            slack_outbound=slack,
            redis_ckpt=ckpt,
            pg_ckpt=ckpt,
        )

        # Mint the session by sending a normal turn first.
        await redis_client.set("session:wecom:ext-susp", "sess-abc", ex=3600)
        await redis_client.set("last_seen:wecom:ext-susp", str(time.time()), ex=3600)
        await redis_client.set("session_status:sess-abc", b"suspended", ex=600)

        await handler(_msg(text="客户继续问问题", user="ext-susp"))

        assert forwarded == [("1700.5", "客户继续问问题")]
        # LLM was NOT called.
        llm_caller.chat.assert_not_called()


class TestInterruptRouting:
    """Phase-2 P0.D: when graph paused at transfer_to_human, run on_interrupt
    instead of dispatching a normal reply."""

    async def test_on_interrupt_invoked_and_normal_reply_skipped(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        # Build a tiny graph that always interrupts.
        from langgraph.graph import END, StateGraph
        from langgraph.types import interrupt

        async def call_interrupt(state):
            interrupt({"type": "transfer_to_human", "reason": "复杂"})
            return state

        ckpt = RedisCheckpointer(redis_client, ttl_seconds=600)
        g = StateGraph(dict)
        g.add_node("n", call_interrupt)
        g.set_entry_point("n")
        g.add_edge("n", END)
        graph = g.compile(checkpointer=ckpt)

        notifier_alerts: list[dict] = []

        class _StubSlack:
            async def get_thread_for_session(self, session_id):
                return None

            async def post_customer_message(self, **kwargs):
                pass

            async def post_handoff_alert(self, **kwargs):
                notifier_alerts.append(kwargs)
                return "ts-1"

        slack = _StubSlack()

        sent: list[tuple[str, str]] = []

        async def fake_send(uid, text):
            sent.append((uid, text))

        cold = RedisCheckpointer(
            fakeredis.aioredis.FakeRedis(decode_responses=False), ttl_seconds=86400
        )

        handler = make_bus_handler(
            graph=graph,
            pool=_fake_pool({"profile": {}}),
            redis=redis_client,
            llm_caller=MagicMock(),
            sends={"wecom": fake_send},
            silence_seconds=1800,
            cache_ttl_seconds=120,
            slack_outbound=slack,
            redis_ckpt=ckpt,
            pg_ckpt=cold,
        )

        await handler(_msg(text="复杂的问题", user="ext-int"))

        # Slack alert posted.
        assert len(notifier_alerts) == 1
        assert notifier_alerts[0]["transfer_reason"] == "复杂"
        # No customer reply was dispatched (graph is paused, no AIMessage yet).
        assert sent == []
