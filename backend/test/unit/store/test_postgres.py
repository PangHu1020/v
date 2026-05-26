"""Unit tests for ``backend.app.store.postgres``.

These tests mock asyncpg. The real-database checks live in
``backend/test/e2e/test_migration.py``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from backend.app.store.postgres import acquire_with_schema, pg_health


class _FakeConn:
    """A mock asyncpg connection that records executed SQL and supports transactions."""

    def __init__(self, fetchval_return: int | Exception = 1) -> None:
        self.executed: list[str] = []
        self._fetchval_return = fetchval_return

    async def execute(self, sql: str, *args: object) -> str:
        self.executed.append(sql)
        return "OK"

    async def fetchval(self, sql: str, *args: object) -> object:
        if isinstance(self._fetchval_return, Exception):
            raise self._fetchval_return
        return self._fetchval_return

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()


class _FakeTransaction:
    async def __aenter__(self) -> _FakeTransaction:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        pass


def _fake_pool(conn: _FakeConn | Exception) -> MagicMock:
    """Build a mock pool whose ``acquire()`` yields ``conn`` (or raises)."""
    pool = MagicMock(spec=asyncpg.Pool)

    @asynccontextmanager
    async def _acquire():
        if isinstance(conn, Exception):
            raise conn
        yield conn

    pool.acquire = _acquire
    return pool


class TestAcquireWithSchema:
    async def test_sets_search_path_for_agent(self) -> None:
        conn = _FakeConn()
        pool = _fake_pool(conn)
        async with acquire_with_schema(pool, "agent") as c:
            assert c is conn
        assert any("search_path = agent" in s for s in conn.executed)

    async def test_sets_search_path_for_dw(self) -> None:
        conn = _FakeConn()
        pool = _fake_pool(conn)
        async with acquire_with_schema(pool, "dw") as c:
            assert c is conn
        assert any("search_path = dw" in s for s in conn.executed)

    async def test_sets_search_path_for_meta(self) -> None:
        conn = _FakeConn()
        pool = _fake_pool(conn)
        async with acquire_with_schema(pool, "meta") as c:
            assert c is conn
        assert any("search_path = meta" in s for s in conn.executed)

    async def test_uses_set_local_inside_transaction(self) -> None:
        conn = _FakeConn()
        pool = _fake_pool(conn)
        async with acquire_with_schema(pool, "agent"):
            pass
        assert "SET LOCAL search_path = agent, public" in conn.executed[0]


class TestPgHealth:
    async def test_returns_true_on_success(self) -> None:
        pool = _fake_pool(_FakeConn(fetchval_return=1))
        assert await pg_health(pool) is True

    async def test_returns_false_on_unexpected_value(self) -> None:
        pool = _fake_pool(_FakeConn(fetchval_return=999))
        assert await pg_health(pool) is False

    async def test_returns_false_on_acquire_exception(self) -> None:
        pool = _fake_pool(RuntimeError("boom"))
        assert await pg_health(pool) is False

    async def test_returns_false_on_fetchval_exception(self) -> None:
        pool = _fake_pool(_FakeConn(fetchval_return=RuntimeError("disconnect")))
        assert await pg_health(pool) is False


class TestCreatePool:
    async def test_create_pool_returns_pool_with_init_callback(self) -> None:
        from backend.app.store.postgres import create_pool

        target = "backend.app.store.postgres.asyncpg.create_pool"
        with patch(target, new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MagicMock(spec=asyncpg.Pool)
            pool = await create_pool("postgresql://x", min_size=2, max_size=8)
            assert pool is mock_create.return_value
            kwargs = mock_create.call_args.kwargs
            assert kwargs["dsn"] == "postgresql://x"
            assert kwargs["min_size"] == 2
            assert kwargs["max_size"] == 8
            assert callable(kwargs["init"])

    async def test_create_pool_raises_when_asyncpg_returns_none(self) -> None:
        from backend.app.store.postgres import create_pool

        target = "backend.app.store.postgres.asyncpg.create_pool"
        with patch(target, new_callable=AsyncMock) as mock_create:
            mock_create.return_value = None
            with pytest.raises(RuntimeError, match="returned None"):
                await create_pool("postgresql://x")
