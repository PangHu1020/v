"""Unit tests for ``backend.v.cron.tasks.consolidate_session`` (Phase-3 reshape).

The consolidator now produces an :class:`ExtractionResult` and dual-writes
into Redis working memory + PG ``agent.event_memory``. These tests stub
the LLMCaller, the embedder, and the PG pool, and use a real
``RedisCheckpointer`` over fakeredis to exercise the message extraction
+ transcript formatting path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import fakeredis.aioredis
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from backend.v.agents.checkpoints.redis import RedisCheckpointer
from backend.v.cron.tasks.consolidate_session import _format_history, consolidate_session
from backend.v.memory.types import ExtractionResult, MemoryEntry
from backend.v.memory.working import read_working_memory
from backend.v.models.llm_caller import LLMResult


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


def _fake_pool(
    *,
    session_row: dict | None = None,
) -> tuple[MagicMock, list[tuple[str, tuple]]]:
    """Mock pool that records every SQL + lets fetchrow return ``session_row``."""
    pool = MagicMock(spec=asyncpg.Pool)
    conn = MagicMock()
    log: list[tuple[str, tuple]] = []

    async def _execute(sql: str, *args):
        log.append((sql, args))
        return "OK"

    async def _fetchrow(sql: str, *args):
        log.append((sql, args))
        if "FROM agent.session WHERE" in sql:
            return session_row
        return None

    conn.execute = _execute
    conn.fetchrow = _fetchrow

    @asynccontextmanager
    async def _txn():
        yield None

    conn.transaction = _txn

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    return pool, log


def _embedder(vectors: list[list[float]] | None = None) -> MagicMock:
    embedder = MagicMock()
    embedder.aembed_documents = AsyncMock(return_value=vectors or [])
    return embedder


def _llm_returning(extracted: ExtractionResult) -> MagicMock:
    caller = MagicMock()
    caller.chat = AsyncMock(
        return_value=LLMResult(
            message=AIMessage(content=extracted.model_dump_json()),
            model="deepseek-flash",
            role="summary",
            fallback_used=False,
            latency_ms=1,
        )
    )
    return caller


async def _seed_thread(redis_client, thread_id: str, messages: list) -> None:
    ckpt = RedisCheckpointer(redis_client, ttl_seconds=1800)
    cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    await ckpt.aput(
        cfg,
        {
            "v": 4,
            "id": "c-1",
            "ts": "2026-05-29T00:00:00+00:00",
            "channel_values": {"messages": messages},
            "channel_versions": {"messages": "1"},
            "versions_seen": {},
            "pending_sends": [],
        },
        {"step": 0},
        {},
    )


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
            "embedder": _embedder(),
        }
        result = await consolidate_session(ctx, session_id="never-existed")
        assert result is None

    async def test_full_round_trip_writes_working_and_events(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        await _seed_thread(
            redis_client,
            "sess-x",
            [
                SystemMessage(content="客服 prompt"),
                HumanMessage(content="我的订单 ORD123 怎么还没到？"),
                AIMessage(content="您好，我帮您查询一下。"),
            ],
        )
        extracted = ExtractionResult(
            profile_updates={},
            working_memories=[
                MemoryEntry(
                    content="客户咨询订单 ORD123 的物流",
                    importance=0.5,
                    keywords=["订单"],
                    kind="event",
                ),
                MemoryEntry(
                    content="客户语气平稳",
                    importance=0.2,
                    keywords=["情绪"],
                    kind="observation",
                ),
            ],
            event_memories=[
                MemoryEntry(
                    content="咨询过订单 ORD123 的配送状态",
                    importance=0.6,
                    keywords=["物流", "ORD123"],
                    kind="event",
                ),
            ],
        )
        pool, log = _fake_pool(session_row={"channel": "wecom", "channel_user_id": "u-1"})
        ctx = {
            "pool": pool,
            "redis": redis_client,
            "llm_caller": _llm_returning(extracted),
            "embedder": _embedder([[0.1] * 1024]),
        }
        result = await consolidate_session(ctx, session_id="sess-x")
        assert result == {"working_inserted": 2, "events_inserted": 1}

        # Working memory landed in Redis.
        working = await read_working_memory(redis_client, session_id="sess-x")
        assert [e.content for e in working] == [
            "客户咨询订单 ORD123 的物流",
            "客户语气平稳",
        ]

        sqls = [sql for sql, _ in log]
        assert any("INSERT INTO agent.event_memory" in s for s in sqls)
        assert any("UPDATE agent.session" in s and "consolidated" in s for s in sqls)

    async def test_resolves_identity_when_caller_omits(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        await _seed_thread(
            redis_client,
            "sess-y",
            [HumanMessage(content="hi"), AIMessage(content="hello")],
        )
        extracted = ExtractionResult(
            profile_updates={},
            working_memories=[],
            event_memories=[
                MemoryEntry(content="说了一声你好", importance=0.1, keywords=[], kind="event")
            ],
        )
        pool, log = _fake_pool(session_row={"channel": "feishu", "channel_user_id": "fu"})
        ctx = {
            "pool": pool,
            "redis": redis_client,
            "llm_caller": _llm_returning(extracted),
            "embedder": _embedder([[0.0] * 1024]),
        }
        result = await consolidate_session(ctx, session_id="sess-y")
        assert result == {"working_inserted": 0, "events_inserted": 1}

        # The identity-lookup SELECT happened.
        assert any("FROM agent.session WHERE" in sql for sql, _ in log)

    async def test_llm_failure_returns_none(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        await _seed_thread(
            redis_client,
            "sess-fail",
            [HumanMessage(content="hi"), AIMessage(content="hello")],
        )
        caller = MagicMock()
        caller.chat = AsyncMock(side_effect=RuntimeError("api down"))

        pool, log = _fake_pool()
        ctx = {
            "pool": pool,
            "redis": redis_client,
            "llm_caller": caller,
            "embedder": _embedder(),
        }
        result = await consolidate_session(ctx, session_id="sess-fail")
        assert result is None
        # No DB writes happened.
        assert not any("INSERT" in sql or "UPDATE" in sql for sql, _ in log)

    async def test_corrupt_json_returns_none(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        await _seed_thread(
            redis_client,
            "sess-bad",
            [HumanMessage(content="x"), AIMessage(content="y")],
        )
        caller = MagicMock()
        caller.chat = AsyncMock(
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
            "llm_caller": caller,
            "embedder": _embedder(),
        }
        result = await consolidate_session(ctx, session_id="sess-bad")
        assert result is None
        assert not any("INSERT" in sql or "UPDATE" in sql for sql, _ in log)
