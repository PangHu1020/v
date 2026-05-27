"""Unit tests for ``backend.v.cron.tasks.repurchase``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import fakeredis.aioredis
import pytest

from backend.v.cron.tasks.repurchase import (
    build_repurchase_targets,
    send_repurchase_reminders,
)


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


def _fake_pool(rows: list[dict]) -> MagicMock:
    pool = MagicMock(spec=asyncpg.Pool)
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    return pool


class TestBuildRepurchaseTargets:
    async def test_maps_rows_to_target_dicts(self) -> None:
        rows = [
            {
                "channel": "wecom",
                "channel_user_id": "ext-1",
                "product_name": "蒙牛纯牛奶",
                "year": 2026,
                "month": 5,
                "day": 12,
            },
            {
                "channel": "feishu",
                "channel_user_id": "ou_42",
                "product_name": "雀巢咖啡",
                "year": 2026,
                "month": 5,
                "day": 6,
            },
        ]
        pool = _fake_pool(rows)
        targets = await build_repurchase_targets(pool)
        assert len(targets) == 2
        assert targets[0]["channel"] == "wecom"
        assert targets[0]["product_name"] == "蒙牛纯牛奶"
        assert targets[0]["order_date"] == "2026-05-12"
        assert targets[1]["order_date"] == "2026-05-06"

    async def test_empty_warehouse_returns_empty_list(self) -> None:
        targets = await build_repurchase_targets(_fake_pool([]))
        assert targets == []


class TestSendRepurchaseReminders:
    async def test_each_target_dispatched(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        sent: list[tuple[str, str]] = []

        async def fake_send(uid: str, text: str) -> None:
            sent.append((uid, text))

        ctx = {"redis": redis_client, "sends": {"wecom": fake_send}}
        targets = [
            {
                "channel": "wecom",
                "channel_user_id": "ext-A",
                "product_name": "蒙牛纯牛奶",
                "order_date": "2026-05-12",
            },
            {
                "channel": "wecom",
                "channel_user_id": "ext-B",
                "product_name": "乐事薯片",
                "order_date": "2026-05-06",
            },
        ]
        summary = await send_repurchase_reminders(ctx, targets=targets)
        assert summary == {"delivered": 2, "skipped": 0}
        assert len(sent) == 2
        assert "蒙牛纯牛奶" in sent[0][1]
        assert "乐事薯片" in sent[1][1]
        assert "2026-05-12" in sent[0][1]

    async def test_skips_unrouted_channel(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        sent: list[tuple[str, str]] = []

        async def fake_send(uid: str, text: str) -> None:
            sent.append((uid, text))

        ctx = {"redis": redis_client, "sends": {"wecom": fake_send}}
        targets = [
            {
                "channel": "wecom",
                "channel_user_id": "u1",
                "product_name": "A",
                "order_date": "2026-05-10",
            },
            {
                "channel": "discord",  # not registered
                "channel_user_id": "u2",
                "product_name": "B",
                "order_date": "2026-05-09",
            },
        ]
        summary = await send_repurchase_reminders(ctx, targets=targets)
        assert summary == {"delivered": 1, "skipped": 1}
        assert len(sent) == 1
