"""Unit tests for ``backend.v.hooks.handoff``.

Exercises the suspend / resume flow with:

- Two ``RedisCheckpointer`` instances (hot + cold) to exercise the
  migration helpers without needing the official Postgres saver in unit
  scope; the e2e test against the real PG saver is in
  ``backend/test/e2e/test_pg_checkpointer.py``.
- A stub ``HandoffNotifier`` that records the handoff alert payload.
- A real LangGraph compiled around an in-test agent that calls
  ``transfer_to_human``, hits the interrupt, and is resumed by
  :func:`on_resume` with collected operator messages.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import asyncpg
import fakeredis.aioredis
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from backend.v.agents.checkpointer import RedisCheckpointer
from backend.v.hooks.handoff import (
    append_operator_log,
    extract_interrupt,
    is_suspended,
    on_interrupt,
    on_resume,
)
from backend.v.tools import transfer_to_human


class _StubNotifier:
    def __init__(self) -> None:
        self.alerts: list[dict] = []

    async def post_handoff_alert(
        self,
        *,
        session_id: str,
        channel: str,
        channel_user_id: str,
        customer_text: str,
        transfer_reason: str,
    ) -> str:
        self.alerts.append(
            {
                "session_id": session_id,
                "channel": channel,
                "channel_user_id": channel_user_id,
                "customer_text": customer_text,
                "transfer_reason": transfer_reason,
            }
        )
        return f"slack-ts-{session_id}"


def _fake_pg_pool(updates: list[tuple[str, str]]) -> MagicMock:
    """A pool whose execute() records (sql, session_id)."""
    pool = MagicMock(spec=asyncpg.Pool)

    @asynccontextmanager
    async def _acquire():
        conn = MagicMock()

        async def _execute(sql: str, *args):
            updates.append((sql, args[0] if args else ""))

        conn.execute = _execute
        yield conn

    pool.acquire = _acquire
    return pool


@pytest.fixture
async def hot() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


@pytest.fixture
async def cold() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


def _build_handoff_graph(checkpointer):  # type: ignore[no-untyped-def]
    """Two-node graph: agent -> tools (interrupt) -> agent -> END.

    The agent first emits a tool call; ``ToolNode`` runs the tool which
    interrupts. After resume the agent issues a final reply.
    """
    from langgraph.prebuilt import ToolNode

    async def _agent(state: dict) -> dict:
        from langchain_core.messages import AIMessage

        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        # If the last message is a tool result, finish.
        if hasattr(last, "tool_call_id"):
            reply = AIMessage(content="按操作员指示回复客户")
            return {"messages": [reply]}
        # Otherwise emit a tool call.
        ai = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "transfer_to_human",
                    "args": {"reason": "复杂退款"},
                }
            ],
        )
        return {"messages": [ai]}

    def _route(state):
        last = state["messages"][-1] if state.get("messages") else None
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "end"

    g = StateGraph(dict)
    g.add_node("agent", _agent)
    g.add_node("tools", ToolNode([transfer_to_human]))
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", _route, {"tools": "tools", "end": END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)


class TestExtractInterrupt:
    async def test_returns_payload_when_paused(
        self,
        hot: fakeredis.aioredis.FakeRedis,
    ) -> None:
        ckpt = RedisCheckpointer(hot, ttl_seconds=600)
        graph = _build_handoff_graph(ckpt)
        cfg: RunnableConfig = {"configurable": {"thread_id": "t-int"}}
        result = await graph.ainvoke({"messages": [HumanMessage("帮我退款")]}, config=cfg)
        payload = await extract_interrupt(result)
        assert payload is not None
        assert payload["type"] == "transfer_to_human"
        assert payload["reason"] == "复杂退款"

    async def test_returns_none_when_completed(
        self,
        hot: fakeredis.aioredis.FakeRedis,
    ) -> None:
        async def _just_finish(state):
            return {"messages": [AIMessage(content="done")]}

        g = StateGraph(dict)
        g.add_node("n", _just_finish)
        g.set_entry_point("n")
        g.add_edge("n", END)
        ckpt = RedisCheckpointer(hot, ttl_seconds=60)
        graph = g.compile(checkpointer=ckpt)
        cfg: RunnableConfig = {"configurable": {"thread_id": "t-done"}}
        result = await graph.ainvoke({"messages": []}, config=cfg)
        assert await extract_interrupt(result) is None

    async def test_returns_none_for_non_dict(self) -> None:
        assert await extract_interrupt(None) is None
        assert await extract_interrupt("string") is None
        assert await extract_interrupt(42) is None


class TestOnInterrupt:
    async def test_migrates_marks_suspended_and_alerts(
        self,
        hot: fakeredis.aioredis.FakeRedis,
        cold: fakeredis.aioredis.FakeRedis,
    ) -> None:
        # Seed a checkpoint in hot.
        hot_ckpt = RedisCheckpointer(hot, ttl_seconds=600)
        cold_ckpt = RedisCheckpointer(cold, ttl_seconds=86400)

        await hot_ckpt.aput(
            {"configurable": {"thread_id": "s-1"}},
            {
                "v": 4,
                "id": "c-1",
                "ts": "now",
                "channel_values": {"foo": "bar"},
                "channel_versions": {"foo": "1"},
                "versions_seen": {},
                "pending_sends": [],
            },
            {"step": 0},
            {},
        )

        notifier = _StubNotifier()
        updates: list[tuple[str, str]] = []
        pool = _fake_pg_pool(updates)

        ts = await on_interrupt(
            session_id="s-1",
            interrupt_payload={"type": "transfer_to_human", "reason": "客户要求"},
            channel="wecom",
            channel_user_id="ext-1",
            customer_text="帮我退款",
            redis=hot,
            pg_pool=pool,
            redis_ckpt=hot_ckpt,
            pg_ckpt=cold_ckpt,
            slack_outbound=notifier,
        )

        # Slack alert posted.
        assert ts == "slack-ts-s-1"
        assert len(notifier.alerts) == 1
        alert = notifier.alerts[0]
        assert alert["session_id"] == "s-1"
        assert alert["transfer_reason"] == "客户要求"

        # Migration happened: hot empty, cold has the checkpoint.
        assert await hot_ckpt.aget_tuple({"configurable": {"thread_id": "s-1"}}) is None
        cold_tuple = await cold_ckpt.aget_tuple({"configurable": {"thread_id": "s-1"}})
        assert cold_tuple is not None

        # Status flipped to suspended in Redis cache.
        assert await is_suspended("s-1", redis=hot)

        # PG was updated.
        assert any("status = 'suspended'" in sql for sql, _ in updates)


class TestAppendOperatorLog:
    async def test_round_trip(self, hot: fakeredis.aioredis.FakeRedis) -> None:
        await append_operator_log("s-X", "operator says hi", redis=hot)
        await append_operator_log("s-X", "follow-up", redis=hot)
        items = await hot.lrange("operator_log:s-X", 0, -1)
        decoded = [m.decode("utf-8") for m in items]
        assert decoded == ["operator says hi", "follow-up"]


class TestOnResume:
    async def test_drains_log_and_invokes_graph(
        self,
        hot: fakeredis.aioredis.FakeRedis,
        cold: fakeredis.aioredis.FakeRedis,
    ) -> None:
        # Production: the graph reads/writes through hot_ckpt. Build it that
        # way and pause the thread there first.
        hot_ckpt = RedisCheckpointer(hot, ttl_seconds=600)
        cold_ckpt = RedisCheckpointer(cold, ttl_seconds=86400)

        graph = _build_handoff_graph(hot_ckpt)
        cfg: RunnableConfig = {"configurable": {"thread_id": "s-resume", "llm_caller": MagicMock()}}
        result = await graph.ainvoke(
            {"messages": [SystemMessage("sys"), HumanMessage("帮我退款")]},
            config=cfg,
        )
        assert await extract_interrupt(result) is not None

        # Simulate the on_interrupt step: migrate hot -> cold so the resume
        # hook later has to pull state back out of cold.
        from backend.v.agents.checkpointer_migration import migrate_hot_to_cold

        await migrate_hot_to_cold("s-resume", redis_ckpt=hot_ckpt, pg_ckpt=cold_ckpt)

        # Pre-load operator messages.
        await append_operator_log("s-resume", "已联系仓库", redis=hot)
        await append_operator_log("s-resume", "请告知客户 24h 内回复", redis=hot)
        await hot.set("session_status:s-resume", b"suspended", ex=600)

        sent: list[tuple[str, str]] = []

        async def fake_send(uid: str, text: str) -> None:
            sent.append((uid, text))

        updates: list[tuple[str, str]] = []
        pool = _fake_pg_pool(updates)

        reply = await on_resume(
            session_id="s-resume",
            channel="wecom",
            channel_user_id="ext-r",
            redis=hot,
            pg_pool=pool,
            redis_ckpt=hot_ckpt,
            pg_ckpt=cold_ckpt,
            graph=graph,
            llm_caller=MagicMock(),
            sends={"wecom": fake_send},
        )

        # Operator log drained.
        remaining = await hot.lrange("operator_log:s-resume", 0, -1)
        assert remaining == []

        # Status flipped to active in PG.
        assert any("status = 'active'" in sql for sql, _ in updates)

        # AI's resumed reply was sent.
        assert reply == "按操作员指示回复客户"
        assert sent == [("ext-r", "按操作员指示回复客户")]


class TestIsSuspended:
    async def test_default_false(self, hot: fakeredis.aioredis.FakeRedis) -> None:
        assert not await is_suspended("none", redis=hot)

    async def test_true_after_set(self, hot: fakeredis.aioredis.FakeRedis) -> None:
        await hot.set("session_status:s-x", b"suspended", ex=60)
        assert await is_suspended("s-x", redis=hot)

    async def test_false_when_active(self, hot: fakeredis.aioredis.FakeRedis) -> None:
        await hot.set("session_status:s-y", b"active", ex=60)
        assert not await is_suspended("s-y", redis=hot)
