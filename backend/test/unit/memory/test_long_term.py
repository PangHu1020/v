"""Unit tests for ``backend.v.memory.long_term`` (mocked PG)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import asyncpg

from backend.v.memory.long_term import read_user_profile


def _fake_pool(row: dict | None) -> MagicMock:
    pool = MagicMock(spec=asyncpg.Pool)
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=row)

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    return pool


class TestReadUserProfile:
    async def test_returns_dict_when_row_present(self) -> None:
        pool = _fake_pool({"profile": {"member_level": "gold", "tier": 2}})
        result = await read_user_profile(pool, channel="wecom", channel_user_id="ext-1")
        assert result == {"member_level": "gold", "tier": 2}

    async def test_returns_none_when_no_row(self) -> None:
        pool = _fake_pool(None)
        result = await read_user_profile(pool, channel="wecom", channel_user_id="missing")
        assert result is None

    async def test_returns_empty_dict_when_profile_null(self) -> None:
        pool = _fake_pool({"profile": None})
        result = await read_user_profile(pool, channel="wecom", channel_user_id="ext-2")
        assert result == {}
