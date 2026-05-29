"""Unit tests for the ``/livez``, ``/readyz``, and legacy ``/health`` endpoints."""

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


@pytest.fixture
async def all_down_client() -> AsyncClient:
    app = _build_app(pg_ok=False, redis_ok=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestLivez:
    async def test_always_ok_when_healthy(self, healthy_client: AsyncClient) -> None:
        resp = await healthy_client.get("/livez")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    async def test_ok_even_when_deps_down(self, all_down_client: AsyncClient) -> None:
        # Liveness must NOT fail on dep outage — restarting the process
        # won't bring Postgres back, so k8s should leave it alone.
        resp = await all_down_client.get("/livez")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestReadyz:
    async def test_ready_when_all_deps_ok(self, healthy_client: AsyncClient) -> None:
        resp = await healthy_client.get("/readyz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "postgres": "ok", "redis": "ok"}

    async def test_503_when_postgres_down(self, degraded_client: AsyncClient) -> None:
        resp = await degraded_client.get("/readyz")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "down"
        assert body["postgres"] == "down"
        assert body["redis"] == "ok"

    async def test_503_when_all_deps_down(self, all_down_client: AsyncClient) -> None:
        resp = await all_down_client.get("/readyz")
        assert resp.status_code == 503
        body = resp.json()
        assert body == {"status": "down", "postgres": "down", "redis": "down"}


class TestHealth:
    async def test_healthy(self, healthy_client: AsyncClient) -> None:
        resp = await healthy_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"status": "ok", "postgres": "ok", "redis": "ok"}

    async def test_partial_failure_reports_down_overall(self, degraded_client: AsyncClient) -> None:
        resp = await degraded_client.get("/health")
        # Legacy /health always returns 200 — failure surfaces only in the body.
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
