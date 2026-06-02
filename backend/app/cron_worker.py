"""ARQ worker entry point.

Run as a separate process from the FastAPI app:

    uv run arq backend.app.cron_worker.WorkerSettings
"""

from __future__ import annotations

from typing import Any, ClassVar

from arq.connections import RedisSettings as ArqRedisSettings
from arq.cron import cron

from backend.app.store import close_client, close_pool, create_client, create_pool
from backend.app.wecom_aibot.outbound import WecomAibotOutbound
from backend.v.configs import get_settings
from backend.v.cron import (
    build_repurchase_targets,
    consolidate_session,
    notify_logistics_delivered,
    push_ad,
    send_repurchase_reminders,
)
from backend.v.memory import extract_session_memory
from backend.v.models.factory import get_embedding
from backend.v.models.llm_caller import LLMCaller
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("app.cron_worker")


async def _scheduled_repurchase_run(ctx: dict[str, Any]) -> dict[str, int]:
    """Daily cron entry: scan for targets and send reminders in one shot."""
    targets = await build_repurchase_targets(ctx["pool"])
    return await send_repurchase_reminders(ctx, targets=targets)


def _arq_redis_settings(url: str) -> ArqRedisSettings:
    return ArqRedisSettings.from_dsn(url)


async def on_startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(level=settings.runtime.log_level, json=settings.runtime.env != "dev")
    _log.info("cron_worker.startup", env=settings.runtime.env)

    pool = await create_pool(settings.db.dsn)
    redis = await create_client(settings.redis.url)

    wecom_aibot_outbound = WecomAibotOutbound(
        redis,
        pubsub_channel=settings.wecom_aibot.outbound_pubsub_channel,
    )

    ctx.update(
        {
            "settings": settings,
            "pool": pool,
            "redis": redis,
            "wecom_aibot_outbound": wecom_aibot_outbound,
            "sends": {
                "wecom_aibot": wecom_aibot_outbound.send_text,
            },
            "llm_caller": LLMCaller(settings.llm),
            "embedder": get_embedding(settings.llm, settings.embedding),
            "ttl_seconds": settings.memory.working_ttl_seconds,
        }
    )


async def on_shutdown(ctx: dict[str, Any]) -> None:
    _log.info("cron_worker.shutdown")
    await close_client(ctx["redis"])
    await close_pool(ctx["pool"])


class WorkerSettings:
    """ARQ worker definition. Pass to ``arq`` CLI as a dotted path."""

    functions: ClassVar = [
        notify_logistics_delivered,
        push_ad,
        consolidate_session,
        extract_session_memory,
        _scheduled_repurchase_run,
    ]
    cron_jobs: ClassVar = [
        cron(_scheduled_repurchase_run, hour={9}, minute={30}),
    ]
    on_startup = on_startup
    on_shutdown = on_shutdown
    max_jobs = 10
    job_timeout = 120

    _settings = get_settings()
    redis_settings = _arq_redis_settings(_settings.arq.effective_redis_url(_settings.redis.url))
