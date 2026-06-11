"""Unit tests for ``backend.v.memory.memory_extractor`` (V2 session-end pipeline)."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import fakeredis.aioredis
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.v.memory.memory_extractor import promote_to_long_term
from backend.v.memory.types import MemoryCandidate, MemoryEntry, MemoryExtraction
from backend.v.memory.working import append_working_memory
from backend.v.models.llm_caller import LLMResult


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


class _FakeDB:
    """Stateful fake covering session lookup + user_memory + episodic insert."""

    def __init__(self, *, session_row: dict | None) -> None:
        self.session_row = session_row
        self.user_rows: list[dict] = []
        self.profile_cache: dict | None = None
        self.episodic_inserts = 0
        self.dedup_existing: list[dict] = []

    async def fetchrow(self, sql: str, *args):
        if "FROM agent.session WHERE" in sql:
            return self.session_row
        if "FROM agent.user_memory" in sql and "FOR UPDATE" in sql:
            for r in self.user_rows:
                if r["attr_key"] == args[2] and r["status"] == "active":
                    return {
                        "id": uuid.UUID(r["id"]),
                        "attr_value": json.dumps(r["attr_value"], ensure_ascii=False),
                        "source": r["source"],
                        "confidence": r["confidence"],
                    }
            return None
        return None

    async def fetch(self, sql: str, *args):
        if "FROM agent.user_memory" in sql and "status = 'active'" in sql:
            return [
                {
                    "id": uuid.UUID(r["id"]),
                    "attr_key": r["attr_key"],
                    "attr_value": json.dumps(r["attr_value"], ensure_ascii=False),
                    "kind": r["kind"],
                    "source": r["source"],
                    "confidence": r["confidence"],
                    "valid_from": None,
                    "last_confirmed_at": None,
                }
                for r in self.user_rows
                if r["status"] == "active"
            ]
        if "FROM agent.event_memory" in sql and "subject = ANY" in sql:
            return self.dedup_existing
        return []

    async def fetchval(self, sql: str, *args):
        if "INSERT INTO agent.user_memory" in sql:
            new_id = str(uuid.uuid4())
            self.user_rows.append(
                {
                    "id": new_id,
                    "attr_key": args[2],
                    "attr_value": json.loads(args[3]),
                    "kind": args[4],
                    "source": args[5],
                    "confidence": args[6],
                    "status": "active",
                }
            )
            return uuid.UUID(new_id)
        return None

    async def execute(self, sql: str, *args):
        if "SET status = 'superseded'" in sql:
            for r in self.user_rows:
                if r["id"] == str(args[0]):
                    r["status"] = "superseded"
        elif "INSERT INTO agent.user_profile" in sql:
            self.profile_cache = json.loads(args[2])
        elif "INSERT INTO agent.event_memory" in sql:
            self.episodic_inserts += 1
        return "OK"


def _pool(db: _FakeDB) -> MagicMock:
    pool = MagicMock(spec=asyncpg.Pool)
    conn = MagicMock()
    conn.fetch = db.fetch
    conn.fetchrow = db.fetchrow
    conn.fetchval = db.fetchval
    conn.execute = db.execute

    @asynccontextmanager
    async def _acquire() -> AsyncIterator[MagicMock]:
        yield conn

    @asynccontextmanager
    async def _transaction() -> AsyncIterator[None]:
        yield None

    conn.transaction = _transaction
    pool.acquire = _acquire
    return pool


def _llm_returning(extraction: MemoryExtraction) -> MagicMock:
    caller = MagicMock()
    caller.chat = AsyncMock(
        return_value=LLMResult(
            message=AIMessage(content=extraction.model_dump_json()),
            model="deepseek-flash",
            role="memory_extract",
            fallback_used=False,
            latency_ms=1,
            parsed=extraction,
        )
    )
    return caller


def _embedder(vectors: list[list[float]]) -> MagicMock:
    e = MagicMock()
    e.aembed_documents = AsyncMock(return_value=vectors)
    return e


async def _seed_working(redis_client, session_id: str, entries: list[MemoryEntry]) -> None:
    await append_working_memory(
        redis_client, session_id=session_id, entries=entries, ttl_seconds=1800
    )


class TestPromoteToLongTerm:
    async def test_no_session_row_returns_none(self, redis_client) -> None:
        db = _FakeDB(session_row=None)
        ctx = {
            "pool": _pool(db),
            "redis": redis_client,
            "llm_caller": MagicMock(),
            "embedder": MagicMock(),
        }
        assert await promote_to_long_term(ctx, session_id="ghost") is None

    async def test_no_input_returns_none(self, redis_client) -> None:
        db = _FakeDB(session_row={"channel": "wecom", "channel_user_id": "u"})
        ctx = {
            "pool": _pool(db),
            "redis": redis_client,
            "llm_caller": MagicMock(),
            "embedder": MagicMock(),
        }
        # No working memory, no fallback messages.
        assert await promote_to_long_term(ctx, session_id="s") is None

    async def test_full_pipeline_user_and_episodic(self, redis_client) -> None:
        db = _FakeDB(session_row={"channel": "wecom", "channel_user_id": "ext-1"})
        await _seed_working(
            redis_client,
            "s",
            [
                MemoryEntry(
                    content="客户偏好顺丰", importance=0.7, keywords=["快递"], kind="preference"
                )
            ],
        )
        extraction = MemoryExtraction(
            user_candidates=[
                MemoryCandidate(
                    content="偏好顺丰",
                    importance=0.7,
                    source="stated",
                    confidence=0.9,
                    attr_key="preferred_courier",
                    attr_value="顺丰",
                    kind="preference",
                )
            ],
            episodic_candidates=[
                MemoryCandidate(content="咨询过 SKU-A 尺码", importance=0.5, subject="SKU-A"),
                MemoryCandidate(content="投诉物流延误", importance=0.8, subject="物流"),
            ],
        )
        ctx = {
            "pool": _pool(db),
            "redis": redis_client,
            "llm_caller": _llm_returning(extraction),
            "embedder": _embedder([[0.1] * 4, [0.2] * 4]),
        }
        result = await promote_to_long_term(ctx, session_id="s")
        assert result == {"user_memory_written": 1, "events_inserted": 2}
        assert db.profile_cache is not None  # rebuilt after user write
        assert db.episodic_inserts == 2
        # Working memory cleared after success.
        assert await redis_client.exists("working_memory:s") == 0

    async def test_importance_floor_drops_trivia(self, redis_client) -> None:
        db = _FakeDB(session_row={"channel": "wecom", "channel_user_id": "u"})
        await _seed_working(
            redis_client, "s", [MemoryEntry(content="x", importance=0.5, kind="event")]
        )
        extraction = MemoryExtraction(
            episodic_candidates=[
                MemoryCandidate(content="琐事", importance=0.1, subject="t"),  # below floor
            ]
        )
        ctx = {
            "pool": _pool(db),
            "redis": redis_client,
            "llm_caller": _llm_returning(extraction),
            "embedder": _embedder([]),
        }
        result = await promote_to_long_term(ctx, session_id="s")
        assert result == {"user_memory_written": 0, "events_inserted": 0}
        assert db.episodic_inserts == 0

    async def test_pii_masked_before_insert(self, redis_client) -> None:
        db = _FakeDB(session_row={"channel": "wecom", "channel_user_id": "u"})
        await _seed_working(
            redis_client, "s", [MemoryEntry(content="x", importance=0.5, kind="event")]
        )
        captured: list[str] = []

        async def _execute(sql, *args):
            if "INSERT INTO agent.event_memory" in sql:
                captured.append(args[3])  # content
                db.episodic_inserts += 1
            return "OK"

        pool = _pool(db)
        # swap execute to capture content
        async with pool.acquire() as conn:
            conn.execute = _execute

        extraction = MemoryExtraction(
            episodic_candidates=[
                MemoryCandidate(content="客户电话 13812341234", importance=0.6, subject="联系")
            ]
        )
        ctx = {
            "pool": pool,
            "redis": redis_client,
            "llm_caller": _llm_returning(extraction),
            "embedder": _embedder([[0.1] * 4]),
        }
        await promote_to_long_term(ctx, session_id="s")
        assert captured and "13812341234" not in captured[0]

    async def test_fallback_transcript_used_when_no_working(self, redis_client) -> None:
        db = _FakeDB(session_row={"channel": "wecom", "channel_user_id": "u"})
        extraction = MemoryExtraction(
            user_candidates=[
                MemoryCandidate(
                    content="偏好京东",
                    importance=0.6,
                    attr_key="preferred_courier",
                    attr_value="京东",
                    kind="preference",
                )
            ]
        )
        ctx = {
            "pool": _pool(db),
            "redis": redis_client,
            "llm_caller": _llm_returning(extraction),
            "embedder": _embedder([]),
            "fallback_messages": [
                HumanMessage(content="我以后都用京东"),
                AIMessage(content="好的，已记录"),
            ],
        }
        result = await promote_to_long_term(ctx, session_id="s")
        assert result == {"user_memory_written": 1, "events_inserted": 0}

    async def test_llm_failure_returns_none(self, redis_client) -> None:
        db = _FakeDB(session_row={"channel": "wecom", "channel_user_id": "u"})
        await _seed_working(
            redis_client, "s", [MemoryEntry(content="x", importance=0.5, kind="event")]
        )
        caller = MagicMock()
        caller.chat = AsyncMock(side_effect=RuntimeError("api down"))
        ctx = {
            "pool": _pool(db),
            "redis": redis_client,
            "llm_caller": caller,
            "embedder": MagicMock(),
        }
        assert await promote_to_long_term(ctx, session_id="s") is None

    async def test_unparseable_output_returns_none(self, redis_client) -> None:
        db = _FakeDB(session_row={"channel": "wecom", "channel_user_id": "u"})
        await _seed_working(
            redis_client, "s", [MemoryEntry(content="x", importance=0.5, kind="event")]
        )
        caller = MagicMock()
        caller.chat = AsyncMock(
            return_value=LLMResult(
                message=AIMessage(content="garbage"),
                model="m",
                role="memory_extract",
                fallback_used=False,
                latency_ms=1,
                parsed=None,  # structured parse failed
            )
        )
        ctx = {
            "pool": _pool(db),
            "redis": redis_client,
            "llm_caller": caller,
            "embedder": MagicMock(),
        }
        assert await promote_to_long_term(ctx, session_id="s") is None
