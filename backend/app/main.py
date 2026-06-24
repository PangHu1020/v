"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from backend.app.bus.consumer import BusConsumer
from backend.app.bus.memory_bus import IdleWatcher, MemoryConsumer
from backend.app.bus.messages import SystemMessage
from backend.app.bus.producer import BusProducer
from backend.app.bus.shard import RedisStreamShard
from backend.app.bus.worker import (
    _consolidate_user_memory,
    _extract_to_working,
    _promote_prev_session,
    make_bus_handler,
)
from backend.app.gateway.middleware import RequestIdMiddleware
from backend.app.gateway.routers import health
from backend.app.store import close_client, close_pool, create_client, create_pool
from backend.v.agents.checkpoints.redis import RedisCheckpointer
from backend.v.agents.graph import build_graph
from backend.v.configs import get_settings, load_agent_config
from backend.v.mcp import MCPRegistry, MCPToolCache
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

    # Extension config (.agent/config.json): MCP servers + skill sources, each
    # enable-gated, secrets via ${VAR}. Missing file → both disabled.
    agent_cfg = load_agent_config()

    skills = []
    for src in agent_cfg.skill_sources:
        skills.extend(load_skills(src.path))
    skill_registry = SkillRegistry(skills)
    if agent_cfg.skill_sources:
        _log.info(
            "app.skills.loaded",
            count=len(skill_registry),
            sources=len(agent_cfg.skill_sources),
        )

    # MCP: connect enabled servers, discover their tools, bind on top of the
    # built-in AGENT_TOOLS. No enabled servers → no registry, no extra tools.
    mcp_registry = None
    mcp_tools: list[Any] = []
    mcp_configs = agent_cfg.mcp_servers
    if mcp_configs:
        mcp_cache = MCPToolCache(
            redis,
            l1_ttl_seconds=settings.mcp.cache_l1_ttl_seconds,
            l2_ttl_seconds=settings.mcp.cache_l2_ttl_seconds,
        )
        mcp_registry = MCPRegistry(
            mcp_configs,
            mcp_cache,
            call_timeout_seconds=settings.mcp.call_timeout_seconds,
        )
        await mcp_registry.connect_all()
        mcp_tools = mcp_registry.tools
        _log.info("app.mcp.ready", servers=len(mcp_configs), tools=len(mcp_tools))

    graph = build_graph(
        redis_ckpt,
        extra_tools=mcp_tools,
        tool_permissions=agent_cfg.tool_permissions,
    )

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
        settings=settings,
        embedder=embedder,
        skill_registry=skill_registry,
        recent_events_to_inject=settings.memory.recent_events_to_inject,
        compression_threshold_tokens=settings.memory.compression_threshold_tokens,
        compression_keep_recent_messages=settings.memory.compression_keep_recent_messages,
        token_model=settings.llm.model,
    )

    consumer = BusConsumer(
        shard,
        group=settings.bus.consumer_group,
        block_ms=5000,
    )
    consumer_task = asyncio.create_task(consumer.run(handler))

    # Memory-task consumers — durable replacements for the old fire-and-forget
    # asyncio.create_task calls in the bus worker.
    _ttl = settings.memory.working_ttl_seconds

    async def _promote_handler(fields: dict) -> None:
        await _promote_prev_session(
            fields["session_id"],
            pool=pg_pool,
            redis=redis,
            llm_caller=llm_caller,
            embedder=embedder,
            checkpointer=redis_ckpt,
            settings=settings,
        )

    async def _consolidate_handler(fields: dict) -> None:
        await _consolidate_user_memory(
            channel=fields["channel"],
            channel_user_id=fields["user_id"],
            pool=pg_pool,
            llm_caller=llm_caller,
            embedder=embedder,
            settings=settings,
        )

    async def _extract_handler(fields: dict) -> None:
        await _extract_to_working(
            fields["session_id"],
            redis=redis,
            llm_caller=llm_caller,
            checkpointer=redis_ckpt,
            cache_ttl_seconds=_ttl,
        )

    mem_consumer = MemoryConsumer(
        redis,
        promote_handler=_promote_handler,
        consolidate_handler=_consolidate_handler,
        extract_handler=_extract_handler,
    )
    idle_watcher = IdleWatcher(redis)
    mem_consumer_task = asyncio.create_task(mem_consumer.run())
    idle_watcher_task = asyncio.create_task(idle_watcher.run())

    app.state.settings = settings
    app.state.pg_pool = pg_pool
    app.state.redis = redis
    app.state.bus_shard = shard
    app.state.bus_producer = producer
    app.state.bus_consumer = consumer
    app.state.bus_consumer_task = consumer_task
    app.state.graph = graph
    app.state.llm_caller = llm_caller
    app.state.mcp_registry = mcp_registry

    try:
        yield
    finally:
        _log.info("app.shutdown")
        await consumer.stop()
        await mem_consumer.stop()
        await idle_watcher.stop()
        for t in (consumer_task, mem_consumer_task, idle_watcher_task):
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        if mcp_registry is not None:
            await mcp_registry.aclose()
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
