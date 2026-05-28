"""End-to-end test for the Phase-2 hot/cold checkpointer migration.

Connects to a real Postgres instance (the running pgvector container by
default), creates a throwaway database, applies migrations, and round-
trips a thread between Redis (hot) and Postgres (cold) using the actual
``AsyncPostgresSaver`` from ``langgraph-checkpoint-postgres``.

Skipped when no PG is reachable.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg
import fakeredis.aioredis
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from backend.v.agents.checkpoints.migration import migrate_cold_to_hot, migrate_hot_to_cold
from backend.v.agents.checkpoints.postgres import open_pg_checkpointer
from backend.v.agents.checkpoints.redis import RedisCheckpointer

PG_TEST_HOST = os.environ.get("PG_TEST_HOST", "localhost")
PG_TEST_PORT = int(os.environ.get("PG_TEST_PORT", "5433"))
PG_TEST_USER = os.environ.get("PG_TEST_USER", "postgres")
PG_TEST_PASSWORD = os.environ.get("PG_TEST_PASSWORD", "postgres")

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_DIR = REPO_ROOT / "scripts" / "sql"

pytestmark = pytest.mark.e2e


async def _admin_connect() -> asyncpg.Connection:
    return await asyncpg.connect(
        host=PG_TEST_HOST,
        port=PG_TEST_PORT,
        user=PG_TEST_USER,
        password=PG_TEST_PASSWORD,
        database="postgres",
    )


async def _pg_reachable() -> bool:
    try:
        conn = await _admin_connect()
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def fresh_database() -> AsyncIterator[str]:
    if not await _pg_reachable():
        pytest.skip(
            f"Postgres not reachable at {PG_TEST_HOST}:{PG_TEST_PORT}; "
            "set PG_TEST_HOST/PG_TEST_PORT or start the pgvector container.",
        )

    name = f"test_migration_{secrets.token_hex(4)}"
    admin = await _admin_connect()
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()

    dsn = f"postgresql://{PG_TEST_USER}:{PG_TEST_PASSWORD}@{PG_TEST_HOST}:{PG_TEST_PORT}/{name}"
    # Apply the agent schema migration so AsyncPostgresSaver.setup() can land
    # its tables under the ``agent`` schema.
    conn = await asyncpg.connect(dsn=dsn)
    try:
        for f in sorted(SQL_DIR.glob("0[01]*.sql")):
            # Only run extensions + agent schema; dw/meta unrelated to this test.
            await conn.execute(f.read_text())
    finally:
        await conn.close()

    try:
        yield dsn
    finally:
        admin = await _admin_connect()
        try:
            await admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = $1 AND pid <> pg_backend_pid()",
                name,
            )
            await admin.execute(f'DROP DATABASE "{name}"')
        finally:
            await admin.close()


@pytest.fixture
async def hot_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


def _config(thread_id: str, parent: str | None = None) -> RunnableConfig:
    cfg: dict = {"thread_id": thread_id, "checkpoint_ns": ""}
    if parent:
        cfg["checkpoint_id"] = parent
    return {"configurable": cfg}


def _checkpoint(cid: str, value: str) -> Checkpoint:
    return {
        "v": 4,
        "id": cid,
        "ts": "2026-05-26T00:00:00+00:00",
        "channel_values": {"foo": value},
        "channel_versions": {"foo": "1"},
        "versions_seen": {},
        "pending_sends": [],
    }


async def _seed_three_checkpoints(ckpt: RedisCheckpointer, thread_id: str) -> None:
    ids = ["c-0", "c-1", "c-2"]
    parents = [None, "c-0", "c-1"]
    for cid, parent in zip(ids, parents, strict=False):
        meta: CheckpointMetadata = {"step": int(cid.split("-")[1])}
        await ckpt.aput(_config(thread_id, parent), _checkpoint(cid, cid), meta, {})


class TestPgCheckpointerSetup:
    async def test_tables_land_in_agent_schema(self, fresh_database: str) -> None:
        async with open_pg_checkpointer(fresh_database):
            pass

        # Verify the LangGraph tables are in agent, not public.
        conn = await asyncpg.connect(fresh_database)
        try:
            rows = await conn.fetch(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_name IN "
                "('checkpoints','checkpoint_blobs','checkpoint_writes','checkpoint_migrations')"
            )
        finally:
            await conn.close()

        assert rows, "expected LangGraph checkpoint tables to be created"
        for row in rows:
            assert row["table_schema"] == "agent", (
                f"{row['table_name']} ended up in {row['table_schema']} (should be 'agent')"
            )


class TestMigrateRoundTripAgainstRealPg:
    async def test_hot_to_cold_to_hot(
        self,
        fresh_database: str,
        hot_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        hot = RedisCheckpointer(hot_redis, ttl_seconds=600)

        async with open_pg_checkpointer(fresh_database) as cold:
            await _seed_three_checkpoints(hot, "thread-rt")
            count = await migrate_hot_to_cold("thread-rt", redis_ckpt=hot, pg_ckpt=cold)
            assert count == 3

            # Hot drained, cold has the latest.
            assert await hot.aget_tuple(_config("thread-rt")) is None
            cold_latest = await cold.aget_tuple(_config("thread-rt"))
            assert cold_latest is not None
            assert cold_latest.checkpoint["id"] == "c-2"

            # Migrate back.
            count_back = await migrate_cold_to_hot("thread-rt", pg_ckpt=cold, redis_ckpt=hot)
            assert count_back == 3
            hot_latest = await hot.aget_tuple(_config("thread-rt"))
            assert hot_latest is not None
            assert hot_latest.checkpoint["id"] == "c-2"
