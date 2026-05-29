"""Unit tests for ``backend.v.memory.memory_extractor``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import fakeredis.aioredis
import pytest
from langchain_core.messages import AIMessage

from backend.v.memory.memory_extractor import promote_to_long_term
from backend.v.memory.types import ExtractionResult, MemoryEntry
from backend.v.memory.working import append_working_memory
from backend.v.models.llm_caller import LLMResult


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


def _fake_pool(
    *,
    session_row: dict | None = None,
    profile_row: dict | None = None,
) -> tuple[MagicMock, list[tuple[str, tuple]]]:
    pool = MagicMock(spec=asyncpg.Pool)
    log: list[tuple[str, tuple]] = []

    async def _fetchrow(sql: str, *args):
        log.append((sql, args))
        if "FROM agent.session WHERE" in sql:
            return session_row
        if "FROM agent.user_profile" in sql:
            return profile_row
        return None

    async def _execute(sql: str, *args):
        log.append((sql, args))
        return "OK"

    conn = MagicMock()
    conn.fetchrow = _fetchrow
    conn.execute = _execute

    @asynccontextmanager
    async def _acquire():
        yield conn

    @asynccontextmanager
    async def _transaction():
        yield None

    conn.transaction = _transaction
    pool.acquire = _acquire
    return pool, log


def _llm_returning(extracted: ExtractionResult) -> MagicMock:
    caller = MagicMock()
    caller.chat = AsyncMock(
        return_value=LLMResult(
            message=AIMessage(content=extracted.model_dump_json()),
            model="deepseek-flash",
            role="memory_extract",
            fallback_used=False,
            latency_ms=1,
        )
    )
    return caller


def _embedder_returning(vectors: list[list[float]]) -> MagicMock:
    embedder = MagicMock()
    embedder.aembed_documents = AsyncMock(return_value=vectors)
    return embedder


async def _seed_working(redis_client, session_id: str, entries: list[MemoryEntry]) -> None:
    await append_working_memory(
        redis_client, session_id=session_id, entries=entries, ttl_seconds=1800
    )


class TestPromoteToLongTerm:
    async def test_no_session_row_returns_none(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        pool, log = _fake_pool(session_row=None)
        ctx = {
            "pool": pool,
            "redis": redis_client,
            "llm_caller": MagicMock(),
            "embedder": MagicMock(),
        }
        result = await promote_to_long_term(ctx, session_id="ghost")
        assert result is None
        assert len(log) == 1  # only the session-identity lookup ran

    async def test_no_working_memory_returns_none(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        pool, _ = _fake_pool(
            session_row={"channel": "wecom", "channel_user_id": "u"},
        )
        ctx = {
            "pool": pool,
            "redis": redis_client,
            "llm_caller": MagicMock(),
            "embedder": MagicMock(),
        }
        result = await promote_to_long_term(ctx, session_id="s")
        assert result is None

    async def test_full_round_trip_writes_profile_and_events(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        pool, log = _fake_pool(
            session_row={"channel": "wecom", "channel_user_id": "ext-1"},
            profile_row={"profile": {"member_level": "黄金"}},
        )
        await _seed_working(
            redis_client,
            "s",
            [
                MemoryEntry(
                    content="客户偏好顺丰快递", importance=0.7, keywords=["快递"], kind="preference"
                ),
            ],
        )
        extracted = ExtractionResult(
            profile_updates={
                "preferred_salutation": "先生",
                "extras": {"preferred_courier": "顺丰"},
            },
            event_memories=[
                MemoryEntry(
                    content="咨询过 SKU-A 的尺码",
                    importance=0.5,
                    keywords=["商品"],
                    kind="event",
                ),
                MemoryEntry(
                    content="投诉过物流延误",
                    importance=0.8,
                    keywords=["物流"],
                    kind="event",
                ),
            ],
        )
        embedder = _embedder_returning([[0.1] * 1024, [0.2] * 1024])
        ctx = {
            "pool": pool,
            "redis": redis_client,
            "llm_caller": _llm_returning(extracted),
            "embedder": embedder,
        }
        result = await promote_to_long_term(ctx, session_id="s")
        assert result == {"profile_updated": 1, "events_inserted": 2}

        sqls = [s for s, _ in log]
        assert any("INSERT INTO agent.user_profile" in s for s in sqls)
        assert sum("INSERT INTO agent.event_memory" in s for s in sqls) == 2

        # Working-memory list deleted after successful promotion.
        assert await redis_client.exists("working_memory:s") == 0

    async def test_empty_profile_updates_skips_upsert(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        pool, log = _fake_pool(
            session_row={"channel": "wecom", "channel_user_id": "u"},
            profile_row={"profile": {"member_level": "黄金"}},
        )
        await _seed_working(
            redis_client,
            "s",
            [MemoryEntry(content="x", importance=0.1, keywords=[], kind="observation")],
        )
        extracted = ExtractionResult(profile_updates={}, event_memories=[])
        ctx = {
            "pool": pool,
            "redis": redis_client,
            "llm_caller": _llm_returning(extracted),
            "embedder": MagicMock(),
        }
        result = await promote_to_long_term(ctx, session_id="s")
        assert result == {"profile_updated": 0, "events_inserted": 0}
        sqls = [s for s, _ in log]
        assert not any("INSERT INTO agent.user_profile" in s for s in sqls)
        assert not any("INSERT INTO agent.event_memory" in s for s in sqls)

    async def test_llm_failure_returns_none(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        pool, _ = _fake_pool(
            session_row={"channel": "wecom", "channel_user_id": "u"},
        )
        await _seed_working(
            redis_client,
            "s",
            [MemoryEntry(content="x", importance=0.1, keywords=[], kind="observation")],
        )
        caller = MagicMock()
        caller.chat = AsyncMock(side_effect=RuntimeError("api down"))
        ctx = {
            "pool": pool,
            "redis": redis_client,
            "llm_caller": caller,
            "embedder": MagicMock(),
        }
        result = await promote_to_long_term(ctx, session_id="s")
        assert result is None

    async def test_corrupt_llm_output_returns_none(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        pool, _ = _fake_pool(
            session_row={"channel": "wecom", "channel_user_id": "u"},
        )
        await _seed_working(
            redis_client,
            "s",
            [MemoryEntry(content="x", importance=0.1, keywords=[], kind="observation")],
        )
        caller = MagicMock()
        caller.chat = AsyncMock(
            return_value=LLMResult(
                message=AIMessage(content="not-json{"),
                model="m",
                role="memory_extract",
                fallback_used=False,
                latency_ms=1,
            )
        )
        ctx = {
            "pool": pool,
            "redis": redis_client,
            "llm_caller": caller,
            "embedder": MagicMock(),
        }
        result = await promote_to_long_term(ctx, session_id="s")
        assert result is None


class TestMemoryEntrySchema:
    def test_importance_clamps(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            MemoryEntry(content="x", importance=2.0, keywords=[], kind="event")

    def test_kind_must_be_known(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            MemoryEntry(content="x", importance=0.5, keywords=[], kind="bogus")  # type: ignore[arg-type]

    def test_round_trip(self) -> None:
        e = MemoryEntry(
            content="客户偏好夜间收货",
            importance=0.8,
            keywords=["快递", "时间"],
            kind="preference",
        )
        rehydrated = MemoryEntry.model_validate_json(e.model_dump_json())
        assert rehydrated == e
