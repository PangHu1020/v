"""Unit tests for ``backend.v.cron.tasks.consolidate_session``.

Stubs the LLMCaller (returns a canned ``SessionSummary`` JSON) and the
PG pool. Uses a real ``RedisCheckpointer`` over fakeredis seeded with a
short conversation to exercise the message extraction + transcript
formatting path.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import fakeredis.aioredis
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from backend.v.agents.checkpointer import RedisCheckpointer
from backend.v.cron.tasks.consolidate_session import (
    SessionSummary,
    _format_history,
    consolidate_session,
)
from backend.v.models.llm_caller import LLMResult


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


def _fake_pool() -> tuple[MagicMock, list[tuple[str, tuple]]]:
    """Mock pool that records (sql, args) for every execute / fetchval."""
    pool = MagicMock(spec=asyncpg.Pool)
    conn = MagicMock()
    log: list[tuple[str, tuple]] = []

    async def _execute(sql: str, *args):
        log.append((sql, args))
        return "OK"

    async def _fetchval(sql: str, *args):
        log.append((sql, args))
        return "row-id-1"

    conn.execute = _execute
    conn.fetchval = _fetchval

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    return pool, log


def _seed_checkpoint_with_messages(
    redis: fakeredis.aioredis.FakeRedis,
    *,
    thread_id: str,
    messages: list,
) -> RedisCheckpointer:
    """Put a single checkpoint into Redis whose channel_values has the messages."""
    ckpt = RedisCheckpointer(redis, ttl_seconds=1800)

    cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    checkpoint = {
        "v": 4,
        "id": "c-1",
        "ts": "2026-05-26T00:00:00+00:00",
        "channel_values": {"messages": messages},
        "channel_versions": {"messages": "1"},
        "versions_seen": {},
        "pending_sends": [],
    }
    return ckpt, cfg, checkpoint


class TestFormatHistory:
    def test_skips_system(self) -> None:
        msgs = [
            SystemMessage(content="rules"),
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
        ]
        text = _format_history(msgs)
        assert "rules" not in text
        assert "客户: hi" in text
        assert "助手: hello" in text

    def test_summarizes_tool_calls(self) -> None:
        ai_with_tools = AIMessage(
            content="",
            tool_calls=[{"id": "1", "name": "transfer_to_human", "args": {"reason": "x"}}],
        )
        msgs = [HumanMessage(content="求人工"), ai_with_tools]
        text = _format_history(msgs)
        assert "transfer_to_human" in text
        assert "[助手调用工具" in text


class TestConsolidateSession:
    async def test_no_checkpoint_returns_none(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        pool, _ = _fake_pool()
        ctx = {
            "pool": pool,
            "redis": redis_client,
            "llm_caller": MagicMock(),
        }
        result = await consolidate_session(ctx, session_id="never-existed")
        assert result is None

    async def test_full_round_trip(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        # Seed a thread with messages.
        ckpt = RedisCheckpointer(redis_client, ttl_seconds=1800)
        cfg: RunnableConfig = {"configurable": {"thread_id": "sess-x"}}
        await ckpt.aput(
            cfg,
            {
                "v": 4,
                "id": "c-1",
                "ts": "2026-05-26T00:00:00+00:00",
                "channel_values": {
                    "messages": [
                        SystemMessage(content="客服 prompt"),
                        HumanMessage(content="我的订单 ORD123 怎么还没到？"),
                        AIMessage(content="您好，我帮您查询一下。"),
                    ]
                },
                "channel_versions": {"messages": "1"},
                "versions_seen": {},
                "pending_sends": [],
            },
            {"step": 0},
            {},
        )

        # Mock LLM to return a structured summary.
        canned = SessionSummary(
            narrative="客户咨询订单 ORD123 的物流状态，助手已开始查询。",
            intents=["物流查询"],
            key_facts=["订单号: ORD123"],
            sentiment="neutral",
            unresolved=["订单到达时间未确认"],
        )
        llm_caller = MagicMock()
        llm_caller.chat = AsyncMock(
            return_value=LLMResult(
                message=AIMessage(content=canned.model_dump_json()),
                model="deepseek-flash",
                role="summary",
                fallback_used=False,
                latency_ms=1,
            )
        )

        pool, log = _fake_pool()
        ctx = {
            "pool": pool,
            "redis": redis_client,
            "llm_caller": llm_caller,
        }
        row_id = await consolidate_session(ctx, session_id="sess-x")
        assert row_id == "row-id-1"

        # Verify both PG ops ran with the right SQL.
        sqls = [sql for sql, _ in log]
        assert any("INSERT INTO agent.session_memory" in s for s in sqls)
        assert any("UPDATE agent.session" in s and "consolidated" in s for s in sqls)

        # Verify the LLM was given the transcript.
        _, kwargs = llm_caller.chat.call_args
        passed_msgs = llm_caller.chat.call_args.args[1]
        human = next(m for m in passed_msgs if isinstance(m, HumanMessage))
        assert "ORD123" in human.content
        assert kwargs["structured"] is SessionSummary

    async def test_llm_failure_returns_none(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        ckpt = RedisCheckpointer(redis_client, ttl_seconds=1800)
        await ckpt.aput(
            {"configurable": {"thread_id": "sess-fail"}},
            {
                "v": 4,
                "id": "c-1",
                "ts": "now",
                "channel_values": {
                    "messages": [HumanMessage(content="hi"), AIMessage(content="hello")]
                },
                "channel_versions": {"messages": "1"},
                "versions_seen": {},
                "pending_sends": [],
            },
            {"step": 0},
            {},
        )

        llm_caller = MagicMock()
        llm_caller.chat = AsyncMock(side_effect=RuntimeError("api down"))

        pool, log = _fake_pool()
        ctx = {
            "pool": pool,
            "redis": redis_client,
            "llm_caller": llm_caller,
        }
        result = await consolidate_session(ctx, session_id="sess-fail")
        assert result is None
        # No PG writes happened.
        assert log == []

    async def test_corrupt_json_returns_none(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        ckpt = RedisCheckpointer(redis_client, ttl_seconds=1800)
        await ckpt.aput(
            {"configurable": {"thread_id": "sess-bad"}},
            {
                "v": 4,
                "id": "c-1",
                "ts": "now",
                "channel_values": {"messages": [HumanMessage(content="x"), AIMessage(content="y")]},
                "channel_versions": {"messages": "1"},
                "versions_seen": {},
                "pending_sends": [],
            },
            {"step": 0},
            {},
        )
        llm_caller = MagicMock()
        llm_caller.chat = AsyncMock(
            return_value=LLMResult(
                message=AIMessage(content="not-json{"),
                model="m",
                role="summary",
                fallback_used=False,
                latency_ms=1,
            )
        )

        pool, log = _fake_pool()
        ctx = {
            "pool": pool,
            "redis": redis_client,
            "llm_caller": llm_caller,
        }
        result = await consolidate_session(ctx, session_id="sess-bad")
        assert result is None
        assert log == []


class TestSessionSummarySchema:
    def test_default_field_values(self) -> None:
        s = SessionSummary(narrative="x")
        assert s.intents == []
        assert s.key_facts == []
        assert s.sentiment == "neutral"
        assert s.unresolved == []

    def test_round_trip_json(self) -> None:
        s = SessionSummary(
            narrative="客户咨询订单",
            intents=["物流查询"],
            key_facts=["ORD123"],
            sentiment="positive",
            unresolved=[],
        )
        rehydrated = SessionSummary.model_validate(json.loads(s.model_dump_json()))
        assert rehydrated == s
