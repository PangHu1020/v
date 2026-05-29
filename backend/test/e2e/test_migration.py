"""End-to-end migration test.

Connects to a real PostgreSQL instance, creates a throwaway database, applies
every SQL file under ``scripts/sql/`` in order, and asserts the resulting
schema is correct (extensions, schemas, table existence, vector dimension,
HNSW index, seeded row counts).

Skipped when no PG is reachable. Connection parameters can be overridden via
``PG_TEST_*`` env vars; defaults target the local ``pgvector/pgvector:pg16``
docker container exposed on port 5433.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg
import pytest

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
    """Create a throwaway database, yield its DSN, drop it afterwards."""
    if not await _pg_reachable():
        pytest.skip(
            f"Postgres not reachable at {PG_TEST_HOST}:{PG_TEST_PORT}; "
            "set PG_TEST_HOST/PG_TEST_PORT or start the pgvector container.",
        )

    name = f"test_agent_{secrets.token_hex(4)}"
    admin = await _admin_connect()
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()

    dsn = f"postgresql://{PG_TEST_USER}:{PG_TEST_PASSWORD}@{PG_TEST_HOST}:{PG_TEST_PORT}/{name}"
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


async def _apply_migrations(dsn: str) -> None:
    files = sorted(SQL_DIR.glob("*.sql"))
    assert files, "expected at least one .sql file under scripts/sql/"
    conn = await asyncpg.connect(dsn=dsn)
    try:
        for f in files:
            await conn.execute(f.read_text())
    finally:
        await conn.close()


class TestMigration:
    async def test_extensions_installed(self, fresh_database: str) -> None:
        await _apply_migrations(fresh_database)
        conn = await asyncpg.connect(fresh_database)
        try:
            rows = await conn.fetch("SELECT extname FROM pg_extension ORDER BY extname")
        finally:
            await conn.close()
        names = {r["extname"] for r in rows}
        assert "vector" in names
        assert "pgcrypto" in names

    async def test_three_schemas_present(self, fresh_database: str) -> None:
        await _apply_migrations(fresh_database)
        conn = await asyncpg.connect(fresh_database)
        try:
            rows = await conn.fetch(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name IN ('agent','dw','meta')"
            )
        finally:
            await conn.close()
        names = {r["schema_name"] for r in rows}
        assert names == {"agent", "dw", "meta"}

    async def test_agent_tables_and_tenant_id_columns(self, fresh_database: str) -> None:
        await _apply_migrations(fresh_database)
        conn = await asyncpg.connect(fresh_database)
        try:
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'agent' ORDER BY table_name"
            )
            tenant_rows = await conn.fetch(
                "SELECT table_name FROM information_schema.columns "
                "WHERE table_schema = 'agent' AND column_name = 'tenant_id' "
                "ORDER BY table_name"
            )
        finally:
            await conn.close()
        agent_tables = {r["table_name"] for r in rows}
        # Phase-3 reshape: session_memory + memory_episodes collapsed
        # into a single event_memory table.
        assert agent_tables >= {
            "user_alias",
            "session",
            "user_profile",
            "event_memory",
        }
        # session_memory + memory_episodes are explicitly removed.
        assert "session_memory" not in agent_tables
        assert "memory_episodes" not in agent_tables
        tenant_tables = {r["table_name"] for r in tenant_rows}
        # Per CLAUDE.md, the long-lived agent tables reserve a nullable tenant_id.
        # event_memory was added in 060 without one — Phase-3 P2 (multi-tenant)
        # will add it; for now the canonical long-term tables suffice.
        assert {"user_alias", "session", "user_profile"} <= tenant_tables

    async def test_event_memory_embedding_is_vector_1024(self, fresh_database: str) -> None:
        await _apply_migrations(fresh_database)
        conn = await asyncpg.connect(fresh_database)
        try:
            row = await conn.fetchrow(
                "SELECT format_type(atttypid, atttypmod) AS coltype "
                "FROM pg_attribute "
                "WHERE attrelid = 'agent.event_memory'::regclass "
                "AND attname = 'embedding'"
            )
        finally:
            await conn.close()
        assert row is not None
        assert row["coltype"] == "vector(1024)"

    async def test_hnsw_index_present(self, fresh_database: str) -> None:
        await _apply_migrations(fresh_database)
        conn = await asyncpg.connect(fresh_database)
        try:
            row = await conn.fetchrow(
                "SELECT indexdef FROM pg_indexes "
                "WHERE schemaname='agent' AND tablename='event_memory' "
                "AND indexname='idx_event_memory_embedding_hnsw'"
            )
        finally:
            await conn.close()
        assert row is not None
        assert "hnsw" in row["indexdef"].lower()
        assert "vector_cosine_ops" in row["indexdef"]

    async def test_dw_seed_row_counts(self, fresh_database: str) -> None:
        await _apply_migrations(fresh_database)
        conn = await asyncpg.connect(fresh_database)
        try:
            counts = {
                t: await conn.fetchval(f"SELECT count(*) FROM dw.{t}")
                for t in ("dim_region", "dim_customer", "dim_product", "dim_date", "fact_order")
            }
        finally:
            await conn.close()
        assert counts == {
            "dim_region": 6,
            "dim_customer": 20,
            "dim_product": 15,
            "dim_date": 90,
            "fact_order": 115,
        }

    async def test_meta_tables_empty_but_present(self, fresh_database: str) -> None:
        await _apply_migrations(fresh_database)
        conn = await asyncpg.connect(fresh_database)
        try:
            counts = {
                t: await conn.fetchval(f"SELECT count(*) FROM meta.{t}")
                for t in ("table_info", "column_info", "metric_info", "column_metric")
            }
        finally:
            await conn.close()
        assert counts == {
            "table_info": 0,
            "column_info": 0,
            "metric_info": 0,
            "column_metric": 0,
        }

    async def test_fact_order_fk_constraints(self, fresh_database: str) -> None:
        await _apply_migrations(fresh_database)
        conn = await asyncpg.connect(fresh_database)
        try:
            rows = await conn.fetch(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = 'dw.fact_order'::regclass "
                "AND contype = 'f' "
                "ORDER BY conname"
            )
        finally:
            await conn.close()
        names = {r["conname"] for r in rows}
        assert names == {
            "fk_order_customer",
            "fk_order_product",
            "fk_order_date",
            "fk_order_region",
        }


class TestStoreHelpersAgainstRealPg:
    """Validate that ``backend.app.store.postgres`` works against a real PG."""

    async def test_acquire_with_schema_and_health(self, fresh_database: str) -> None:
        from backend.app.store.postgres import acquire_with_schema, create_pool, pg_health

        await _apply_migrations(fresh_database)
        pool = await create_pool(fresh_database, min_size=1, max_size=2)
        try:
            assert await pg_health(pool) is True

            async with acquire_with_schema(pool, "dw") as conn:
                # Unqualified table reference works because search_path is set.
                count = await conn.fetchval("SELECT count(*) FROM dim_region")
                assert count == 6

            async with acquire_with_schema(pool, "agent") as conn:
                # Round-trip a vector to verify the codec.
                vec = [0.1] * 1024
                await conn.execute(
                    "INSERT INTO event_memory "
                    "(channel, channel_user_id, content, kind, importance, embedding) "
                    "VALUES ($1, $2, $3, $4, $5, $6)",
                    "wecom",
                    "ext-1",
                    "hello",
                    "event",
                    0.5,
                    vec,
                )
                row = await conn.fetchrow("SELECT content, embedding FROM event_memory LIMIT 1")
                assert row["content"] == "hello"
                assert len(row["embedding"]) == 1024
        finally:
            await pool.close()
