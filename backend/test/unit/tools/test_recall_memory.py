"""Unit tests for ``backend.v.tools.recall_memory``.

The tool's SQL talks to pgvector, so unit-level tests focus on the
configuration / formatting paths; the real query is exercised in
``backend/test/e2e/test_recall_memory.py`` against the running pgvector
container.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.v.tools.recall_memory import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    _format_episodes,
    recall_memory,
)


class TestFormatEpisodes:
    def test_empty_returns_placeholder(self) -> None:
        assert _format_episodes([]) == "（暂无相关历史记忆）"

    def test_today_label(self) -> None:
        out = _format_episodes([{"age_days": 0.3, "content": "客户偏好顺丰", "similarity": 0.9}])
        assert "今天" in out
        assert "客户偏好顺丰" in out

    def test_days_ago_label(self) -> None:
        out = _format_episodes([{"age_days": 5.7, "content": "曾投诉物流", "similarity": 0.8}])
        assert "5 天前" in out

    def test_multiple_lines(self) -> None:
        out = _format_episodes(
            [
                {"age_days": 1.0, "content": "A", "similarity": 0.9},
                {"age_days": 10.0, "content": "B", "similarity": 0.7},
            ]
        )
        assert out.count("- ") == 2


class TestRecallMemoryToolConfig:
    async def test_missing_context_returns_friendly_message(self) -> None:
        # No configurable provided.
        out = await recall_memory.ainvoke(
            {"query": "顺丰"},
            config={"configurable": {}},
        )
        assert "无法访问历史记忆" in out

    async def test_missing_partial_context_still_friendly(self) -> None:
        out = await recall_memory.ainvoke(
            {"query": "顺丰"},
            config={
                "configurable": {
                    "pg_pool": MagicMock(),
                    # no embedder / channel
                }
            },
        )
        assert "无法访问" in out

    async def test_top_k_bounded_below(self) -> None:
        # Tool runs but with top_k clamped to 1; we can verify by mocking
        # the pool's execute signature args.
        pool = MagicMock()
        recorded: list[tuple[str, tuple]] = []

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _acquire():
            conn = MagicMock()

            async def _fetch(sql, *args):
                recorded.append((sql, args))
                return []  # empty result -> short-circuits the UPDATE

            async def _execute(sql, *args):
                recorded.append((sql, args))
                return "OK"

            conn.fetch = _fetch
            conn.execute = _execute
            yield conn

        pool.acquire = _acquire

        embedder = MagicMock()
        embedder.aembed_query = AsyncMock(return_value=[0.0] * 1024)

        await recall_memory.ainvoke(
            {"query": "顺丰", "top_k": 0},
            config={
                "configurable": {
                    "pg_pool": pool,
                    "embedder": embedder,
                    "channel": "wecom",
                    "channel_user_id": "ext-1",
                }
            },
        )
        # The bounded top_k is the LAST positional arg of the fetch SELECT.
        select_call = next(c for c in recorded if "SELECT" in c[0])
        # args layout: (vector, channel, user, candidate_pool, half_life_days, top_k)
        assert select_call[1][-1] == 1

    async def test_top_k_bounded_above(self) -> None:
        pool = MagicMock()
        recorded: list[tuple[str, tuple]] = []

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _acquire():
            conn = MagicMock()

            async def _fetch(sql, *args):
                recorded.append((sql, args))
                return []

            conn.fetch = _fetch
            conn.execute = AsyncMock(return_value="OK")
            yield conn

        pool.acquire = _acquire

        embedder = MagicMock()
        embedder.aembed_query = AsyncMock(return_value=[0.0] * 1024)

        await recall_memory.ainvoke(
            {"query": "顺丰", "top_k": 9999},
            config={
                "configurable": {
                    "pg_pool": pool,
                    "embedder": embedder,
                    "channel": "wecom",
                    "channel_user_id": "ext-1",
                }
            },
        )
        select_call = next(c for c in recorded if "SELECT" in c[0])
        assert select_call[1][-1] == MAX_TOP_K


class TestDefaults:
    def test_default_top_k(self) -> None:
        assert DEFAULT_TOP_K == 5

    def test_default_half_life(self) -> None:
        assert DEFAULT_HALF_LIFE_DAYS == 30

    def test_max_top_k(self) -> None:
        assert MAX_TOP_K == 20


@pytest.mark.parametrize(
    "age_days,expected",
    [(0.5, "今天"), (1.0, "1 天前"), (29.7, "29 天前"), (365.0, "365 天前")],
)
def test_age_label(age_days: float, expected: str) -> None:
    out = _format_episodes([{"age_days": age_days, "content": "x", "similarity": 0.5}])
    assert expected in out
