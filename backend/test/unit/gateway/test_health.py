"""Unit tests for the ``/health`` endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.app.gateway.middleware import RequestIdMiddleware
from backend.app.gateway.routers import health


def _build_app(*, pg_ok: bool, redis_ok: bool) -> FastAPI:
    """Build an app with mocked store handles. Avoids the real lifespan."""
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    app.include_router(health.router)
    app.state.pg_pool = MagicMock()
    app.state.redis = MagicMock()

    async def fake_pg_health(_pool: object) -> bool:
        return pg_ok

    async def fake_redis_health(_client: object) -> bool:
        return redis_ok

    health.pg_health = AsyncMock(side_effect=fake_pg_health)  # type: ignore[assignment]
    health.redis_health = AsyncMock(side_effect=fake_redis_health)  # type: ignore[assignment]
    return app


@pytest.fixture
async def healthy_client() -> AsyncClient:
    app = _build_app(pg_ok=True, redis_ok=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def degraded_client() -> AsyncClient:
    app = _build_app(pg_ok=False, redis_ok=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestHealth:
    async def test_healthy(self, healthy_client: AsyncClient) -> None:
        resp = await healthy_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"status": "ok", "postgres": "ok", "redis": "ok"}

    async def test_partial_failure_reports_down_overall(self, degraded_client: AsyncClient) -> None:
        resp = await degraded_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "down"
        assert body["postgres"] == "down"
        assert body["redis"] == "ok"

    async def test_response_carries_request_id_header(self, healthy_client: AsyncClient) -> None:
        resp = await healthy_client.get("/health")
        assert resp.headers.get("x-request-id")

    async def test_response_echoes_inbound_request_id(self, healthy_client: AsyncClient) -> None:
        resp = await healthy_client.get("/health", headers={"X-Request-Id": "abc-123"})
        assert resp.headers.get("x-request-id") == "abc-123"
