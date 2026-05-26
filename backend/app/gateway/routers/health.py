"""Liveness / readiness endpoint reporting Postgres and Redis connectivity."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from backend.app.store import pg_health, redis_health

Status = Literal["ok", "down"]


class HealthResponse(BaseModel):
    status: Status
    postgres: Status
    redis: Status


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return overall service health and per-dependency status.

    Reads pool / client handles from ``request.app.state``. The endpoint never
    raises; failure modes turn into ``"down"`` strings so probes can scrape
    the JSON instead of relying on HTTP status codes.
    """
    pg = "ok" if await pg_health(request.app.state.pg_pool) else "down"
    rd = "ok" if await redis_health(request.app.state.redis) else "down"
    overall: Status = "ok" if pg == "ok" and rd == "ok" else "down"
    return HealthResponse(status=overall, postgres=pg, redis=rd)
