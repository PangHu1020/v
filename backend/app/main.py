"""FastAPI application entrypoint.

Owns the lifespan: opens the asyncpg pool and Redis client on startup,
publishes them on ``app.state``, and closes them on shutdown. Mounts the
gateway routers (currently just ``/health``); customer-channel routers and
the bus consumer task are wired in later phases.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.gateway.middleware import RequestIdMiddleware
from backend.app.gateway.routers import health
from backend.app.store import close_client, close_pool, create_client, create_pool
from backend.v.configs import get_settings
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage process-wide resources for the FastAPI app."""
    settings = get_settings()
    configure_logging(level=settings.runtime.log_level, json=settings.runtime.env != "dev")
    _log.info("app.startup", env=settings.runtime.env)

    pg_pool = await create_pool(settings.db.dsn)
    redis = await create_client(settings.redis.url)

    app.state.settings = settings
    app.state.pg_pool = pg_pool
    app.state.redis = redis

    try:
        yield
    finally:
        _log.info("app.shutdown")
        await close_client(redis)
        await close_pool(pg_pool)


def create_app() -> FastAPI:
    """Construct the FastAPI app. Kept as a factory so tests can build their own."""
    app = FastAPI(
        title="v-agent-platform",
        description=("External customer-service agent platform over WeCom + Feishu (Phase-1)."),
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(health.router)
    return app


app = create_app()
