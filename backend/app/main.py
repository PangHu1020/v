"""FastAPI application entrypoint.

Owns the lifespan: opens the asyncpg pool and Redis client on startup,
constructs the WeCom 智能机器人 outbound adapter, spawns the bus consumer
task wired to the LangGraph turn handler, mounts the gateway / operator
routers, and tears everything down on shutdown.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI

from backend.app.bus.consumer import BusConsumer
from backend.app.bus.messages import SystemMessage
from backend.app.bus.producer import BusProducer
from backend.app.bus.shard import RedisStreamShard
from backend.app.bus.worker import make_bus_handler
from backend.app.gateway.middleware import RequestIdMiddleware
from backend.app.gateway.routers import health
from backend.app.operator.slack.outbound import SlackOutbound
from backend.app.operator.slack.router import build_router as build_slack_router
from backend.app.store import close_client, close_pool, create_client, create_pool
from backend.v.agents.checkpoints.postgres import open_pg_checkpointer
from backend.v.agents.checkpoints.redis import RedisCheckpointer
from backend.v.agents.graph import build_graph
from backend.v.configs import get_settings
from backend.v.cron.tasks.consolidate_session import consolidate_session
from backend.v.hooks.handoff import append_operator_log, on_resume
from backend.v.models.factory import get_embedding
from backend.v.models.llm_caller import LLMCaller
from backend.v.skills import SkillRegistry, load_skills
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger
from backend.v.utils.tracing import configure_langsmith

_log = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage process-wide resources for the FastAPI app."""
    settings = get_settings()
    configure_logging(level=settings.runtime.log_level, json=settings.runtime.env != "dev")
    configure_langsmith(settings.langsmith)
    _log.info("app.startup", env=settings.runtime.env, slack_enabled=settings.slack.enabled)

    pg_pool = await create_pool(settings.db.dsn)
    redis = await create_client(settings.redis.url)

    shard = RedisStreamShard(
        redis,
        prefix=settings.bus.stream_prefix,
        shard_count=settings.bus.shard_count,
    )
    producer = BusProducer(shard)

    async def channel_to_bus(message: SystemMessage) -> None:
        await producer.enqueue(message)

    redis_ckpt = RedisCheckpointer(redis, ttl_seconds=settings.memory.working_ttl_seconds)
    llm_caller = LLMCaller(settings.llm)
    embedder = get_embedding(settings.llm, settings.embedding)

    skill_registry = SkillRegistry(load_skills(settings.skill.internal_repo_path))
    if settings.skill.internal_repo_path:
        _log.info("app.skills.loaded", count=len(skill_registry))

    graph = build_graph(redis_ckpt)

    from backend.app.wecom_aibot.outbound import WecomAibotOutbound

    wecom_aibot_outbound = WecomAibotOutbound(
        redis,
        pubsub_channel=settings.wecom_aibot.outbound_pubsub_channel,
    )
    sends: dict[str, Any] = {"wecom_aibot": wecom_aibot_outbound.send_text}

    exit_stack = AsyncExitStack()
    slack_outbound: SlackOutbound | None = None
    pg_ckpt = None

    if settings.slack.enabled:
        pg_ckpt = await exit_stack.enter_async_context(open_pg_checkpointer(settings.db.dsn))

        slack_outbound = SlackOutbound(
            bot_token=settings.slack.bot_token,
            alert_channel_id=settings.slack.handoff_channel_id,
            redis=redis,
        )

        async def _slack_on_resume(session_id: str) -> None:
            await on_resume(
                session_id=session_id,
                channel=await _channel_for_session(pg_pool, session_id),
                channel_user_id=await _channel_user_id_for_session(pg_pool, session_id),
                redis=redis,
                pg_pool=pg_pool,
                redis_ckpt=redis_ckpt,
                pg_ckpt=pg_ckpt,
                graph=graph,
                llm_caller=llm_caller,
                sends=sends,
            )

        async def _slack_on_operator_message(session_id: str, text: str) -> None:
            await append_operator_log(session_id, text, redis=redis)
            channel = await _channel_for_session(pg_pool, session_id)
            channel_user_id = await _channel_user_id_for_session(pg_pool, session_id)
            send = sends.get(channel)
            if send:
                await send(channel_user_id, text)

        app.include_router(
            build_slack_router(
                signing_secret=settings.slack.signing_secret,
                redis=redis,
                on_resume=_slack_on_resume,
                on_operator_message=_slack_on_operator_message,
            )
        )

    consolidate_ctx = {
        "pool": pg_pool,
        "redis": redis,
        "llm_caller": llm_caller,
        "ttl_seconds": settings.memory.working_ttl_seconds,
        "event_ttl_days": settings.memory.event_ttl_days,
    }

    handler = make_bus_handler(
        graph=graph,
        pool=pg_pool,
        redis=redis,
        llm_caller=llm_caller,
        sends=sends,
        silence_seconds=settings.memory.working_ttl_seconds,
        cache_ttl_seconds=settings.memory.working_ttl_seconds,
        slack_outbound=slack_outbound,
        redis_ckpt=redis_ckpt,
        pg_ckpt=pg_ckpt,
        embedder=embedder,
        skill_registry=skill_registry,
        skill_top_k=settings.skill.max_skills_per_turn,
        recent_events_to_inject=settings.memory.recent_events_to_inject,
        compression_threshold_tokens=settings.memory.compression_threshold_tokens,
        compression_keep_recent_messages=settings.memory.compression_keep_recent_messages,
        token_model=settings.llm.main_primary,
        consolidate_callable=consolidate_session,
        consolidate_ctx=consolidate_ctx,
    )

    consumer = BusConsumer(
        shard,
        group=settings.bus.consumer_group,
        block_ms=5000,
    )
    consumer_task = asyncio.create_task(consumer.run(handler))

    app.state.settings = settings
    app.state.pg_pool = pg_pool
    app.state.redis = redis
    app.state.bus_shard = shard
    app.state.bus_producer = producer
    app.state.bus_consumer = consumer
    app.state.bus_consumer_task = consumer_task
    app.state.slack_outbound = slack_outbound
    app.state.pg_ckpt = pg_ckpt
    app.state.graph = graph
    app.state.llm_caller = llm_caller

    try:
        yield
    finally:
        _log.info("app.shutdown")
        await consumer.stop()
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass
        if slack_outbound is not None:
            await slack_outbound.aclose()
        await exit_stack.aclose()
        await close_client(redis)
        await close_pool(pg_pool)


async def _channel_for_session(pool, session_id: str) -> str:
    """Look up the channel slug for a given session id."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT channel FROM agent.session WHERE session_id = $1",
            session_id,
        )
    return str(row["channel"]) if row else ""


async def _channel_user_id_for_session(pool, session_id: str) -> str:
    """Look up the per-channel user id for a given session id."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT channel_user_id FROM agent.session WHERE session_id = $1",
            session_id,
        )
    return str(row["channel_user_id"]) if row else ""


def create_app() -> FastAPI:
    """Construct the FastAPI app. Kept as a factory so tests can build their own."""
    app = FastAPI(
        title="v-agent-platform",
        description="External customer-service agent platform over WeCom 智能机器人 WebSocket.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(health.router)
    return app


app = create_app()
