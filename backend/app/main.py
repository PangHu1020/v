"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from backend.app.bus.consumer import BusConsumer
from backend.app.bus.messages import SystemMessage
from backend.app.bus.producer import BusProducer
from backend.app.bus.shard import RedisStreamShard
from backend.app.bus.worker import make_bus_handler
from backend.app.gateway.middleware import RequestIdMiddleware
from backend.app.gateway.routers import health
from backend.app.store import close_client, close_pool, create_client, create_pool
from backend.v.agents.checkpoints.redis import RedisCheckpointer
from backend.v.agents.graph import build_graph
from backend.v.configs import get_settings
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
    _log.info("app.startup", env=settings.runtime.env)

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

    from backend.app.channels.wecom_aibot.outbound import WecomAibotOutbound

    wecom_aibot_outbound = WecomAibotOutbound(
        redis,
        pubsub_channel=settings.wecom_aibot.outbound_pubsub_channel,
    )
    sends: dict[str, Any] = {"wecom_aibot": wecom_aibot_outbound.send_text}

    handler = make_bus_handler(
        graph=graph,
        pool=pg_pool,
        redis=redis,
        llm_caller=llm_caller,
        sends=sends,
        silence_seconds=settings.memory.working_ttl_seconds,
        cache_ttl_seconds=settings.memory.working_ttl_seconds,
        embedder=embedder,
        skill_registry=skill_registry,
        skill_top_k=settings.skill.max_skills_per_turn,
        recent_events_to_inject=settings.memory.recent_events_to_inject,
        compression_threshold_tokens=settings.memory.compression_threshold_tokens,
        compression_keep_recent_messages=settings.memory.compression_keep_recent_messages,
        token_model=settings.llm.main_primary,
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
        await close_client(redis)
        await close_pool(pg_pool)


def create_app() -> FastAPI:
    """Construct the FastAPI app."""
    app = FastAPI(
        title="v-agent-platform",
        description="External customer-service agent platform over WeCom 智能机器人 WebSocket.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(health.router)
    return app


app = create_app()
