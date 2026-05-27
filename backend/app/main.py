"""FastAPI application entrypoint.

Owns the lifespan: opens the asyncpg pool and Redis client on startup,
constructs the per-channel adapter chain (debouncer + producer + outbound),
optionally constructs the Slack operator adapter and durable Postgres
checkpointer for Phase-2 handoff, spawns the bus consumer task wired to
the LangGraph turn handler, mounts the gateway / channel / operator
routers, and tears everything down on shutdown.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI

from backend.app.bus.consumer import BusConsumer
from backend.app.bus.messages import SystemMessage
from backend.app.bus.producer import BusProducer
from backend.app.bus.shard import RedisStreamShard
from backend.app.bus.worker import make_bus_handler
from backend.app.channels.debounce import Debouncer
from backend.app.channels.feishu.crypto import FeishuCrypto
from backend.app.channels.feishu.outbound import FeishuOutbound
from backend.app.channels.feishu.router import build_router as build_feishu_router
from backend.app.channels.wecom.crypto import WecomCrypto
from backend.app.channels.wecom.outbound import WecomOutbound
from backend.app.channels.wecom.router import build_router as build_wecom_router
from backend.app.gateway.middleware import RequestIdMiddleware
from backend.app.gateway.routers import health
from backend.app.operator.slack.outbound import SlackOutbound
from backend.app.operator.slack.router import build_router as build_slack_router
from backend.app.store import close_client, close_pool, create_client, create_pool
from backend.v.agents.checkpointer import RedisCheckpointer
from backend.v.agents.graph import build_graph
from backend.v.agents.pg_checkpointer import open_pg_checkpointer
from backend.v.configs import get_settings
from backend.v.hooks.handoff import append_operator_log, on_resume
from backend.v.mcp import MCPRegistry, MCPToolCache, parse_servers
from backend.v.models.llm_caller import LLMCaller
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage process-wide resources for the FastAPI app."""
    settings = get_settings()
    configure_logging(level=settings.runtime.log_level, json=settings.runtime.env != "dev")
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

    debouncer = Debouncer(
        redis,
        window_ms=settings.bus.debounce_ms,
        dispatch=channel_to_bus,
    )

    wecom_crypto = WecomCrypto(settings.wecom.aes_key, settings.wecom.corp_id)
    feishu_crypto = FeishuCrypto(settings.feishu.encrypt_key)

    wecom_outbound = WecomOutbound(
        corp_id=settings.wecom.corp_id,
        secret=settings.wecom.secret,
        agent_id=settings.wecom.agent_id,
    )
    feishu_outbound = FeishuOutbound(
        app_id=settings.feishu.app_id,
        app_secret=settings.feishu.app_secret,
    )

    redis_ckpt = RedisCheckpointer(redis, ttl_seconds=settings.memory.working_ttl_seconds)
    llm_caller = LLMCaller(settings.llm)

    # Phase-2 P1: MCP registry. Tools discovered here are bound to the
    # agent at graph build time, alongside transfer_to_human.
    mcp_cache = MCPToolCache(
        redis,
        l1_ttl_seconds=settings.mcp.cache_l1_ttl_seconds,
        l2_ttl_seconds=settings.mcp.cache_l2_ttl_seconds,
    )
    mcp_configs = parse_servers(settings.mcp.servers_json)
    mcp_registry = MCPRegistry(
        mcp_configs,
        mcp_cache,
        call_timeout_seconds=settings.mcp.call_timeout_seconds,
    )
    if mcp_configs:
        await mcp_registry.connect_all()
        _log.info("app.mcp.connected", tool_count=len(mcp_registry.tools))

    graph = build_graph(redis_ckpt, extra_tools=mcp_registry.tools)

    sends = {
        "wecom": wecom_outbound.send_text,
        "feishu": feishu_outbound.send_text,
    }

    # Phase-2 P0: Slack operator adapter + durable Postgres checkpointer
    # for handoff. Wired in lazily so deployments without Slack credentials
    # keep the Phase-1 single-loop flow.
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
                # We don't know channel/channel_user_id at the Slack callback
                # site (only session_id). The on_resume hook needs them so
                # it can dispatch the AI's reply via the right channel
                # adapter. We resolve them by reading the agent.session row.
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
            # Persist for the resume payload AND relay back to the customer
            # so they see the operator's response in real time.
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
    )

    consumer = BusConsumer(
        shard,
        group=settings.bus.consumer_group,
        block_ms=5000,
    )
    consumer_task = asyncio.create_task(consumer.run(handler))

    # Publish handles for routers + tests.
    app.state.settings = settings
    app.state.pg_pool = pg_pool
    app.state.redis = redis
    app.state.bus_shard = shard
    app.state.bus_producer = producer
    app.state.bus_consumer = consumer
    app.state.bus_consumer_task = consumer_task
    app.state.debouncer = debouncer
    app.state.wecom_outbound = wecom_outbound
    app.state.feishu_outbound = feishu_outbound
    app.state.slack_outbound = slack_outbound
    app.state.pg_ckpt = pg_ckpt
    app.state.graph = graph
    app.state.llm_caller = llm_caller
    app.state.mcp_registry = mcp_registry

    # Mount channel routers now that adapters are constructed.
    app.include_router(build_wecom_router(settings.wecom, wecom_crypto, debouncer))
    app.include_router(build_feishu_router(settings.feishu, feishu_crypto, debouncer))

    try:
        yield
    finally:
        _log.info("app.shutdown")
        await consumer.stop()
        await debouncer.shutdown()
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass
        if slack_outbound is not None:
            await slack_outbound.aclose()
        await wecom_outbound.aclose()
        await feishu_outbound.aclose()
        await mcp_registry.aclose()
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
        description=(
            "External customer-service agent platform over WeCom + Feishu (Phase-1) "
            "with Slack-mediated human handoff (Phase-2 P0)."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(health.router)
    return app


app = create_app()
