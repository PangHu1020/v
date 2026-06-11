"""Unit tests for ``backend.v.memory.consolidation`` (monthly forgetting)."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest
from langchain_core.messages import AIMessage

from backend.v.memory.consolidation import activation, consolidate_user
from backend.v.models.llm_caller import LLMResult


class TestActivation:
    def test_importance_dominates_when_fresh_unaccessed(self) -> None:
        a = activation(importance=0.8, access_count=0, age_days=0.0)
        assert a == pytest.approx(0.8)

    def test_access_reinforces(self) -> None:
        low = activation(importance=0.2, access_count=0, age_days=10.0)
        high = activation(importance=0.2, access_count=20, age_days=10.0)
        assert high > low  # use-it: recall bumps survival

    def test_age_decays(self) -> None:
        fresh = activation(importance=0.5, access_count=1, age_days=1.0)
        old = activation(importance=0.5, access_count=1, age_days=400.0)
        assert old < fresh


class _FakeConsolidationDB:
    """Stateful fake for the consolidation queries."""

    def __init__(self, clusters: dict[tuple[str, str], list[dict]]) -> None:
        # clusters keyed by (period, subject) → list of raw row dicts
        self.clusters = clusters
        self.summaries_inserted: list[dict] = []
        self.deleted_ids: list = []
        self.linked_ids: list = []

    async def fetch(self, sql: str, *args):
        if "GROUP BY period, subject" in sql:
            return [{"period": p, "subject": s} for (p, s) in self.clusters]
        if "EXTRACT(EPOCH" in sql:  # fetch cluster rows
            period, subject = args[2], args[3]
            return [
                {
                    "id": r["id"],
                    "content": r["content"],
                    "importance": r["importance"],
                    "access_count": r["access_count"],
                    "age_days": r["age_days"],
                }
                for r in self.clusters.get((period, subject), [])
            ]
        return []

    async def fetchval(self, sql: str, *args):
        if "INSERT INTO agent.event_memory" in sql and "'summary'" in sql:
            sid = uuid.uuid4()
            self.summaries_inserted.append({"id": sid, "content": args[2], "subject": args[5]})
            return sid
        return None

    async def execute(self, sql: str, *args):
        if "DELETE FROM agent.event_memory" in sql:
            self.deleted_ids.extend(args[0])
        elif "SET consolidated_into" in sql:
            self.linked_ids.extend(args[0])
        return "OK"


def _pool(db: _FakeConsolidationDB) -> MagicMock:
    pool = MagicMock(spec=asyncpg.Pool)
    conn = MagicMock()
    conn.fetch = db.fetch
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


def _llm_summary(summary: str, importance: float = 0.6) -> MagicMock:
    caller = MagicMock()
    caller.chat = AsyncMock(
        return_value=LLMResult(
            message=AIMessage(content=f'{{"summary": "{summary}", "importance": {importance}}}'),
            model="m",
            role="memory_extract",
            fallback_used=False,
            latency_ms=1,
        )
    )
    return caller


def _embedder() -> MagicMock:
    e = MagicMock()
    e.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return e


def _raw(content: str, *, importance: float, access: int, age: float) -> dict:
    return {
        "id": uuid.uuid4(),
        "content": content,
        "importance": importance,
        "access_count": access,
        "age_days": age,
    }


class TestConsolidateUser:
    async def test_no_clusters_returns_none(self) -> None:
        db = _FakeConsolidationDB({})
        ctx = {"pool": _pool(db), "llm_caller": MagicMock(), "embedder": MagicMock()}
        assert await consolidate_user(ctx, channel="wecom", channel_user_id="u") is None

    async def test_summarises_and_prunes_low_activation(self) -> None:
        # Cluster of 3 raws under subject SO123: one fresh+important survives,
        # two old+trivial+unaccessed get deleted.
        cluster = {
            ("2026-05", "SO123"): [
                _raw("催物流1", importance=0.1, access=0, age=300.0),  # low act → delete
                _raw("催物流2", importance=0.1, access=0, age=300.0),  # low act → delete
                _raw("承诺补偿", importance=0.9, access=5, age=2.0),  # high act → keep
            ]
        }
        db = _FakeConsolidationDB(cluster)
        ctx = {
            "pool": _pool(db),
            "llm_caller": _llm_summary("就 SO123 多次催物流，已承诺补偿"),
            "embedder": _embedder(),
            "settings": None,
        }
        result = await consolidate_user(ctx, channel="wecom", channel_user_id="u")
        assert result["summaries"] == 1
        assert result["raws_deleted"] == 2
        assert result["raws_kept"] == 1
        assert len(db.summaries_inserted) == 1
        assert len(db.deleted_ids) == 2
        assert len(db.linked_ids) == 1

    async def test_single_row_cluster_skipped(self) -> None:
        # min_cluster_size default = 2, so a lone raw isn't summarised.
        cluster = {("2026-05", "SO9"): [_raw("一次性事件", importance=0.5, access=0, age=100.0)]}
        db = _FakeConsolidationDB(cluster)
        ctx = {
            "pool": _pool(db),
            "llm_caller": _llm_summary("x"),
            "embedder": _embedder(),
        }
        assert await consolidate_user(ctx, channel="wecom", channel_user_id="u") is None
        assert db.summaries_inserted == []

    async def test_llm_failure_skips_cluster(self) -> None:
        cluster = {
            ("2026-05", "S"): [
                _raw("a", importance=0.5, access=0, age=100.0),
                _raw("b", importance=0.5, access=0, age=100.0),
            ]
        }
        db = _FakeConsolidationDB(cluster)
        caller = MagicMock()
        caller.chat = AsyncMock(side_effect=RuntimeError("down"))
        ctx = {"pool": _pool(db), "llm_caller": caller, "embedder": _embedder()}
        # No summary produced → overall None, nothing deleted.
        assert await consolidate_user(ctx, channel="wecom", channel_user_id="u") is None
        assert db.deleted_ids == []
