"""Unit tests for ``backend.v.memory.memory_extractor``."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest
from langchain_core.messages import AIMessage

from backend.v.memory.memory_extractor import (
    Episode,
    ExtractionOutput,
    extract_session_memory,
)
from backend.v.models.llm_caller import LLMResult


def _fake_pool(
    *,
    session_row: dict | None = None,
    session_memory_row: dict | None = None,
    profile_row: dict | None = None,
) -> tuple[MagicMock, list[tuple[str, tuple]]]:
    """Pool that records SQL and returns canned rows by SQL keyword."""
    pool = MagicMock(spec=asyncpg.Pool)
    log: list[tuple[str, tuple]] = []

    async def _fetchrow(sql: str, *args):
        log.append((sql, args))
        if "FROM agent.session WHERE" in sql:
            return session_row
        if "FROM agent.session_memory" in sql:
            return session_memory_row
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

    pool.acquire = _acquire
    return pool, log


def _llm_returning(extracted: ExtractionOutput) -> MagicMock:
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


class TestExtractSessionMemory:
    async def test_no_session_row_returns_none(self) -> None:
        pool, log = _fake_pool(session_row=None)
        ctx = {"pool": pool, "llm_caller": MagicMock(), "embedder": MagicMock()}
        result = await extract_session_memory(ctx, session_id="ghost")
        assert result is None
        # Only the lookup ran; no LLM, no embed.
        assert len(log) == 1

    async def test_no_session_memory_returns_none(self) -> None:
        pool, _ = _fake_pool(
            session_row={"channel": "wecom", "channel_user_id": "u"},
            session_memory_row=None,
        )
        ctx = {"pool": pool, "llm_caller": MagicMock(), "embedder": MagicMock()}
        result = await extract_session_memory(ctx, session_id="s")
        assert result is None

    async def test_full_round_trip_writes_profile_and_episodes(self) -> None:
        pool, log = _fake_pool(
            session_row={"channel": "wecom", "channel_user_id": "ext-1"},
            session_memory_row={
                "summary": "客户咨询订单 ORD123 物流。",
                "metadata": {
                    "intents": ["物流查询"],
                    "key_facts": ["ORD123"],
                    "sentiment": "neutral",
                    "unresolved": [],
                },
            },
            profile_row={"profile": {"member_level": "黄金"}},
        )
        extracted = ExtractionOutput(
            profile={"member_level": "黄金", "preferred_courier": "顺丰"},
            episodes=[
                Episode(
                    content="客户偏好顺丰快递",
                    importance=4,
                    tags=["preference"],
                ),
                Episode(
                    content="客户对物流速度敏感",
                    importance=3,
                    tags=["sensitivity"],
                ),
            ],
        )
        embedder = _embedder_returning([[0.1] * 1024, [0.2] * 1024])
        ctx = {
            "pool": pool,
            "llm_caller": _llm_returning(extracted),
            "embedder": embedder,
        }
        result = await extract_session_memory(ctx, session_id="s")
        assert result == {"profile_updated": 1, "episodes_inserted": 2}

        sqls = [s for s, _ in log]
        assert any("INSERT INTO agent.user_profile" in s for s in sqls)
        # Two episodes -> two INSERTs.
        assert sum("INSERT INTO agent.memory_episodes" in s for s in sqls) == 2

    async def test_empty_profile_skips_upsert(self) -> None:
        pool, log = _fake_pool(
            session_row={"channel": "wecom", "channel_user_id": "u"},
            session_memory_row={"summary": "x", "metadata": {}},
            profile_row={"profile": {"member_level": "黄金"}},
        )
        # LLM returns empty profile (nothing changed) and no episodes.
        extracted = ExtractionOutput(profile={}, episodes=[])
        ctx = {
            "pool": pool,
            "llm_caller": _llm_returning(extracted),
            "embedder": MagicMock(),
        }
        result = await extract_session_memory(ctx, session_id="s")
        assert result == {"profile_updated": 0, "episodes_inserted": 0}
        sqls = [s for s, _ in log]
        # No INSERT/UPDATE on user_profile or memory_episodes.
        assert not any("INSERT INTO agent.user_profile" in s for s in sqls)
        assert not any("INSERT INTO agent.memory_episodes" in s for s in sqls)

    async def test_llm_failure_returns_none(self) -> None:
        pool, _ = _fake_pool(
            session_row={"channel": "wecom", "channel_user_id": "u"},
            session_memory_row={"summary": "x", "metadata": {}},
            profile_row=None,
        )
        caller = MagicMock()
        caller.chat = AsyncMock(side_effect=RuntimeError("api down"))
        ctx = {"pool": pool, "llm_caller": caller, "embedder": MagicMock()}
        result = await extract_session_memory(ctx, session_id="s")
        assert result is None

    async def test_corrupt_llm_output_returns_none(self) -> None:
        pool, _ = _fake_pool(
            session_row={"channel": "wecom", "channel_user_id": "u"},
            session_memory_row={"summary": "x", "metadata": {}},
            profile_row=None,
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
        ctx = {"pool": pool, "llm_caller": caller, "embedder": MagicMock()}
        result = await extract_session_memory(ctx, session_id="s")
        assert result is None


class TestEpisodeSchema:
    def test_validation_clamps_importance(self) -> None:
        # Out of range raises.
        with pytest.raises(Exception):  # noqa: B017
            Episode(content="x", importance=10)

    def test_default_tags_empty(self) -> None:
        e = Episode(content="只是一个事实")
        assert e.tags == []
        assert e.importance == 3

    def test_round_trip(self) -> None:
        e = Episode(content="客户偏好夜间收货", importance=5, tags=["preference", "logistics"])
        rehydrated = Episode.model_validate(json.loads(e.model_dump_json()))
        assert rehydrated == e
