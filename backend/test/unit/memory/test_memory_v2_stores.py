"""Unit tests for memory V2 leaf stores: user_memory + policy_gate + episodic insert."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import asyncpg

from backend.v.memory.event_memory import insert_episodic_candidates
from backend.v.memory.policy_gate import (
    apply_importance_floor,
    apply_pii_mask,
    dedup_episodic,
)
from backend.v.memory.types import MemoryCandidate
from backend.v.memory.user_memory import (
    project_profile,
    read_active_user_memory,
    rebuild_profile_cache,
    upsert_user_memory,
)


class _FakeUserMemoryDB:
    """Stateful in-memory stand-in for agent.user_memory + agent.user_profile.

    Interprets the handful of SQL statements user_memory.py issues by substring
    match, mutating an in-memory row list so the three-step supersede protocol
    is exercised for real.
    """

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.profile_cache: dict | None = None

    def _active(self, channel: str, user: str, attr_key: str) -> dict | None:
        for r in self.rows:
            if (
                r["channel"] == channel
                and r["channel_user_id"] == user
                and r["attr_key"] == attr_key
                and r["status"] == "active"
            ):
                return r
        return None

    async def fetch(self, sql: str, *args):
        if "FROM agent.user_memory" in sql and "status = 'active'" in sql:
            channel, user = args[0], args[1]
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
                for r in self.rows
                if r["channel"] == channel
                and r["channel_user_id"] == user
                and r["status"] == "active"
            ]
        return []

    async def fetchrow(self, sql: str, *args):
        if "FROM agent.user_memory" in sql and "FOR UPDATE" in sql:
            channel, user, attr_key = args[0], args[1], args[2]
            r = self._active(channel, user, attr_key)
            if r is None:
                return None
            return {
                "id": uuid.UUID(r["id"]),
                "attr_value": json.dumps(r["attr_value"], ensure_ascii=False),
                "source": r["source"],
                "confidence": r["confidence"],
            }
        return None

    async def fetchval(self, sql: str, *args):
        if "INSERT INTO agent.user_memory" in sql:
            new_id = str(uuid.uuid4())
            self.rows.append(
                {
                    "id": new_id,
                    "channel": args[0],
                    "channel_user_id": args[1],
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
            old_id = str(args[0])
            for r in self.rows:
                if r["id"] == old_id:
                    r["status"] = "superseded"
        elif "SET last_confirmed_at = now()" in sql:
            old_id = str(args[0])
            new_conf = args[1]
            new_source = args[2]
            for r in self.rows:
                if r["id"] == old_id:
                    r["confidence"] = max(r["confidence"], new_conf)
                    if new_source == "stated":
                        r["source"] = "stated"
        elif "INSERT INTO agent.user_profile" in sql:
            self.profile_cache = json.loads(args[2])
        return "OK"


def _pool_for(db: _FakeUserMemoryDB) -> MagicMock:
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


def _sem(attr_key: str, value, *, source="inferred", confidence=0.5, kind="preference"):
    return MemoryCandidate(
        content=f"{attr_key}={value}",
        importance=0.6,
        source=source,
        confidence=confidence,
        attr_key=attr_key,
        attr_value=value,
        kind=kind,
    )


class TestUpsertUserMemory:
    async def test_first_value_inserts(self) -> None:
        db = _FakeUserMemoryDB()
        pool = _pool_for(db)
        outcome = await upsert_user_memory(
            pool,
            channel="wecom",
            channel_user_id="u1",
            candidate=_sem("preferred_courier", "顺丰"),
            session_id=None,
        )
        assert outcome == "inserted"
        assert len([r for r in db.rows if r["status"] == "active"]) == 1

    async def test_same_value_reaffirms(self) -> None:
        db = _FakeUserMemoryDB()
        pool = _pool_for(db)
        await upsert_user_memory(
            pool,
            channel="wecom",
            channel_user_id="u1",
            candidate=_sem("preferred_courier", "顺丰", confidence=0.5),
            session_id=None,
        )
        outcome = await upsert_user_memory(
            pool,
            channel="wecom",
            channel_user_id="u1",
            candidate=_sem("preferred_courier", "顺丰", source="stated", confidence=0.9),
            session_id=None,
        )
        assert outcome == "reaffirmed"
        active = [r for r in db.rows if r["status"] == "active"]
        assert len(active) == 1
        assert active[0]["confidence"] == 0.9  # lifted
        assert active[0]["source"] == "stated"

    async def test_conflicting_value_supersedes_when_new_wins(self) -> None:
        db = _FakeUserMemoryDB()
        pool = _pool_for(db)
        await upsert_user_memory(
            pool,
            channel="wecom",
            channel_user_id="u1",
            candidate=_sem("preferred_courier", "顺丰", source="inferred", confidence=0.5),
            session_id=None,
        )
        outcome = await upsert_user_memory(
            pool,
            channel="wecom",
            channel_user_id="u1",
            candidate=_sem("preferred_courier", "京东", source="stated", confidence=0.9),
            session_id=None,
        )
        assert outcome == "superseded"
        active = [r for r in db.rows if r["status"] == "active"]
        superseded = [r for r in db.rows if r["status"] == "superseded"]
        assert len(active) == 1 and active[0]["attr_value"] == "京东"
        assert len(superseded) == 1 and superseded[0]["attr_value"] == "顺丰"

    async def test_weaker_conflicting_value_skipped(self) -> None:
        db = _FakeUserMemoryDB()
        pool = _pool_for(db)
        await upsert_user_memory(
            pool,
            channel="wecom",
            channel_user_id="u1",
            candidate=_sem("preferred_courier", "顺丰", source="stated", confidence=0.9),
            session_id=None,
        )
        outcome = await upsert_user_memory(
            pool,
            channel="wecom",
            channel_user_id="u1",
            candidate=_sem("preferred_courier", "京东", source="inferred", confidence=0.4),
            session_id=None,
        )
        assert outcome == "skipped"
        active = [r for r in db.rows if r["status"] == "active"]
        assert len(active) == 1 and active[0]["attr_value"] == "顺丰"

    async def test_candidate_without_attr_key_skipped(self) -> None:
        db = _FakeUserMemoryDB()
        pool = _pool_for(db)
        outcome = await upsert_user_memory(
            pool,
            channel="wecom",
            channel_user_id="u1",
            candidate=MemoryCandidate(content="x", importance=0.5),
            session_id=None,
        )
        assert outcome == "skipped"


class TestProjectProfile:
    def test_canonical_and_extras(self) -> None:
        rows = [
            {"attr_key": "member_level", "attr_value": "黄金"},
            {"attr_key": "preferred_courier", "attr_value": "顺丰"},
        ]
        profile = project_profile(rows)
        assert profile["member_level"] == "黄金"
        # preferred_courier is not canonical → extras
        assert profile["extras"]["preferred_courier"] == "顺丰"

    def test_list_field(self) -> None:
        rows = [{"attr_key": "risk_flags", "attr_value": ["易投诉", "高价值"]}]
        profile = project_profile(rows)
        assert profile["risk_flags"] == ["易投诉", "高价值"]

    def test_empty(self) -> None:
        assert project_profile([]) == {}


class TestRebuildProfileCache:
    async def test_writes_projected_profile(self) -> None:
        db = _FakeUserMemoryDB()
        pool = _pool_for(db)
        await upsert_user_memory(
            pool,
            channel="wecom",
            channel_user_id="u1",
            candidate=_sem("member_level", "黄金"),
            session_id=None,
        )
        profile = await rebuild_profile_cache(pool, channel="wecom", channel_user_id="u1")
        assert profile["member_level"] == "黄金"
        assert db.profile_cache == profile

    async def test_read_active_roundtrip(self) -> None:
        db = _FakeUserMemoryDB()
        pool = _pool_for(db)
        await upsert_user_memory(
            pool,
            channel="wecom",
            channel_user_id="u1",
            candidate=_sem("preferred_courier", "顺丰"),
            session_id=None,
        )
        active = await read_active_user_memory(pool, channel="wecom", channel_user_id="u1")
        assert len(active) == 1
        assert active[0]["attr_value"] == "顺丰"


class TestPolicyGate:
    def test_importance_floor(self) -> None:
        cands = [
            MemoryCandidate(content="a", importance=0.2),
            MemoryCandidate(content="b", importance=0.5),
        ]
        kept = apply_importance_floor(cands, floor=0.3)
        assert [c.content for c in kept] == ["b"]

    def test_pii_mask_content_and_value(self) -> None:
        cands = [
            MemoryCandidate(content="客户电话 13812341234", importance=0.5),
            _sem("phone", "13812341234"),
        ]
        masked = apply_pii_mask(cands)
        assert "13812341234" not in masked[0].content
        assert "13812341234" not in masked[1].attr_value

    async def test_dedup_drops_similar_same_subject(self) -> None:
        # Existing row with subject 'SO123' has vector [1,0,0]; candidate near-identical.
        async def _fetch(sql, *args):
            return [{"subject": "SO123", "embedding": [1.0, 0.0, 0.0]}]

        pool = MagicMock(spec=asyncpg.Pool)
        conn = MagicMock()
        conn.fetch = _fetch

        @asynccontextmanager
        async def _acquire():
            yield conn

        pool.acquire = _acquire

        cands = [MemoryCandidate(content="重复事件", importance=0.5, subject="SO123")]
        kept, vecs = await dedup_episodic(
            pool,
            channel="wecom",
            channel_user_id="u1",
            candidates=cands,
            vectors=[[1.0, 0.0, 0.0]],
            similarity_floor=0.95,
        )
        assert kept == []
        assert vecs == []

    async def test_dedup_keeps_dissimilar(self) -> None:
        async def _fetch(sql, *args):
            return [{"subject": "SO123", "embedding": [1.0, 0.0, 0.0]}]

        pool = MagicMock(spec=asyncpg.Pool)
        conn = MagicMock()
        conn.fetch = _fetch

        @asynccontextmanager
        async def _acquire():
            yield conn

        pool.acquire = _acquire

        cands = [MemoryCandidate(content="新事件", importance=0.5, subject="SO123")]
        kept, _vecs = await dedup_episodic(
            pool,
            channel="wecom",
            channel_user_id="u1",
            candidates=cands,
            vectors=[[0.0, 1.0, 0.0]],
            similarity_floor=0.95,
        )
        assert len(kept) == 1

    async def test_dedup_no_subject_kept(self) -> None:
        pool = MagicMock(spec=asyncpg.Pool)
        cands = [MemoryCandidate(content="无主题事件", importance=0.5)]
        kept, _vecs = await dedup_episodic(
            pool,
            channel="wecom",
            channel_user_id="u1",
            candidates=cands,
            vectors=[[0.1, 0.2, 0.3]],
            similarity_floor=0.95,
        )
        assert len(kept) == 1


class TestInsertEpisodicCandidates:
    async def test_inserts_with_v2_columns(self) -> None:
        captured: list[tuple] = []

        async def _execute(sql, *args):
            captured.append((sql, args))
            return "OK"

        pool = MagicMock(spec=asyncpg.Pool)
        conn = MagicMock()
        conn.execute = _execute

        @asynccontextmanager
        async def _acquire():
            yield conn

        @asynccontextmanager
        async def _transaction():
            yield None

        conn.transaction = _transaction
        pool.acquire = _acquire

        cands = [
            MemoryCandidate(content="投诉物流", importance=0.6, subject="物流", source="stated")
        ]
        n = await insert_episodic_candidates(
            pool,
            channel="wecom",
            channel_user_id="u1",
            session_id="s1",
            candidates=cands,
            vectors=[[0.1, 0.2]],
        )
        assert n == 1
        sql, args = captured[0]
        assert "tier" in sql and "'raw'" in sql
        assert "物流" in args  # subject passed
        assert "stated" in args  # source passed

    async def test_empty_noop(self) -> None:
        pool = MagicMock(spec=asyncpg.Pool)
        n = await insert_episodic_candidates(
            pool,
            channel="wecom",
            channel_user_id="u1",
            session_id=None,
            candidates=[],
            vectors=[],
        )
        assert n == 0
