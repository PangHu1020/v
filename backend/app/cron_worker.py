"""ARQ worker entry point (Phase-2 P2).

Run as a separate process from the FastAPI app:

    uv run arq backend.app.cron_worker.WorkerSettings

Lives under ``/app/`` because it imports both ``/app/`` channel adapters
(to dispatch proactive messages) and ``/v/cron`` task definitions. Core
task logic stays in ``/v/`` per the layering rule; this module is the
only place where the two layers meet for the cron pathway, mirroring
how :mod:`backend.app.main` is the meeting point for the reactive
pathway.

The worker connects to Postgres + Redis on startup, builds the channel
outbound clients + LLMCaller, packs them into the ARQ context dict, and
hands them to the per-task functions on each invocation.
"""

from __future__ import annotations

from typing import Any, ClassVar

from arq.connections import RedisSettings as ArqRedisSettings
from arq.cron import cron

from backend.app.channels.feishu.outbound import FeishuOutbound
from backend.app.channels.wecom.outbound import WecomOutbound
from backend.app.store import close_client, close_pool, create_client, create_pool
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
    """Translate a ``redis://host:port/db`` URL into ARQ's RedisSettings."""
    # ARQ provides a parser; using it keeps us in sync with future URL changes.
    return ArqRedisSettings.from_dsn(url)


async def on_startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(level=settings.runtime.log_level, json=settings.runtime.env != "dev")
    _log.info("cron_worker.startup", env=settings.runtime.env)

    pool = await create_pool(settings.db.dsn)
    redis = await create_client(settings.redis.url)

    wecom_outbound = WecomOutbound(
        corp_id=settings.wecom.corp_id,
        secret=settings.wecom.secret,
        agent_id=settings.wecom.agent_id,
    )
    feishu_outbound = FeishuOutbound(
        app_id=settings.feishu.app_id,
        app_secret=settings.feishu.app_secret,
    )

    ctx.update(
        {
            "settings": settings,
            "pool": pool,
            "redis": redis,
            "wecom_outbound": wecom_outbound,
            "feishu_outbound": feishu_outbound,
            "sends": {
                "wecom": wecom_outbound.send_text,
                "feishu": feishu_outbound.send_text,
            },
            "llm_caller": LLMCaller(settings.llm),
            "embedder": get_embedding(settings.llm, settings.embedding),
            "ttl_seconds": settings.memory.working_ttl_seconds,
        }
    )


async def on_shutdown(ctx: dict[str, Any]) -> None:
    _log.info("cron_worker.shutdown")
    await ctx["wecom_outbound"].aclose()
    await ctx["feishu_outbound"].aclose()
    await close_client(ctx["redis"])
    await close_pool(ctx["pool"])


class WorkerSettings:
    """ARQ worker definition. Pass to ``arq`` CLI as a dotted path.

    ``functions`` are on-demand jobs you can enqueue from anywhere via
    ``redis_pool.enqueue_job("notify_logistics_delivered", ...)``.

    ``cron_jobs`` run on a schedule. Phase-2 P2 ships one daily run of
    repurchase reminders; logistics + ad push are event-driven and
    enqueued from elsewhere (the bus worker or an admin endpoint).
    """

    functions: ClassVar = [
        notify_logistics_delivered,
        push_ad,
        consolidate_session,
        extract_session_memory,
        _scheduled_repurchase_run,
    ]
    cron_jobs: ClassVar = [
        # 09:30 every day — well after the 09:00 batch jobs hit but before
        # most customers start their afternoon session. The off-zero minute
        # is deliberate to avoid the API stampede that lands on every :00.
        cron(_scheduled_repurchase_run, hour={9}, minute={30}),
    ]
    on_startup = on_startup
    on_shutdown = on_shutdown
    max_jobs = 10
    job_timeout = 120

    @classmethod
    def redis_settings(cls) -> ArqRedisSettings:
        settings = get_settings()
        return _arq_redis_settings(settings.arq.effective_redis_url(settings.redis.url))
