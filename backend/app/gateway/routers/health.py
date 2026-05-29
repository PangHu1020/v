"""Liveness, readiness, and legacy combined health endpoints.

The semantic split matters for k8s / nomad probe configuration:

- ``/livez`` — the process is up and able to serve HTTP. It returns
  ``200`` unconditionally; failing this means the process should be
  killed and restarted. Dependency outages do NOT fail liveness, because
  restarting a healthy process won't bring Postgres back.
- ``/readyz`` — the process is ready to take traffic. It checks every
  upstream dependency (Postgres, Redis); a failing dep flips the
  endpoint to HTTP ``503`` so the load balancer pulls the instance out
  of rotation while leaving the process running.
- ``/health`` — legacy combined endpoint. Always ``200`` with a JSON
  body summarizing each dep. Kept for back-compat with anything already
  scraping it; new probes should target ``/livez`` / ``/readyz``.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from backend.app.store import pg_health, redis_health

Status = Literal["ok", "down"]


class HealthResponse(BaseModel):
    status: Status
    postgres: Status
    redis: Status


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    status: Status
    postgres: Status
    redis: Status


router = APIRouter(tags=["health"])


@router.get("/livez", response_model=LivenessResponse)
async def livez() -> LivenessResponse:
    """Always-ok liveness probe.

    The process being able to answer at all is the signal. Dependencies
    are deliberately NOT checked here — a transient Postgres outage
    should not cause k8s to crash-loop the pod.
    """
    return LivenessResponse()


@router.get(
    "/readyz",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
)
async def readyz(request: Request, response: Response) -> ReadinessResponse:
    """Readiness probe. ``503`` if any required dep is down.

    The body always carries the per-dep breakdown so probes can scrape
    JSON instead of relying solely on the status code.
    """
    pg: Status = "ok" if await pg_health(request.app.state.pg_pool) else "down"
    rd: Status = "ok" if await redis_health(request.app.state.redis) else "down"
    overall: Status = "ok" if pg == "ok" and rd == "ok" else "down"
    if overall == "down":
        response.status_code = 503
    return ReadinessResponse(status=overall, postgres=pg, redis=rd)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Legacy combined endpoint. Kept for back-compat.

    Always returns HTTP ``200`` regardless of dep state — failure modes
    surface as ``"down"`` strings in the body. New probes should use
    ``/livez`` and ``/readyz`` instead.
    """
    pg: Status = "ok" if await pg_health(request.app.state.pg_pool) else "down"
    rd: Status = "ok" if await redis_health(request.app.state.redis) else "down"
    overall: Status = "ok" if pg == "ok" and rd == "ok" else "down"
    return HealthResponse(status=overall, postgres=pg, redis=rd)
