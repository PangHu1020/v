"""FastAPI application entrypoint.

Owns the lifespan: opens the asyncpg pool and Redis client on startup,
constructs the per-channel adapter chain (debouncer + producer + outbound),
spawns the bus consumer task wired to the LangGraph turn handler, mounts
the gateway + channel routers, and tears everything down on shutdown.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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
from backend.app.store import close_client, close_pool, create_client, create_pool
from backend.v.agents.checkpointer import RedisCheckpointer
from backend.v.agents.graph import build_graph
from backend.v.configs import get_settings
from backend.v.models.llm_caller import LLMCaller
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

    checkpointer = RedisCheckpointer(redis, ttl_seconds=settings.memory.working_ttl_seconds)
    graph = build_graph(checkpointer)

    llm_caller = LLMCaller(settings.llm)

    handler = make_bus_handler(
        graph=graph,
        pool=pg_pool,
        redis=redis,
        llm_caller=llm_caller,
        sends={
            "wecom": wecom_outbound.send_text,
            "feishu": feishu_outbound.send_text,
        },
        silence_seconds=settings.memory.working_ttl_seconds,
        cache_ttl_seconds=settings.memory.working_ttl_seconds,
    )

    consumer = BusConsumer(
        shard,
        group=settings.bus.consumer_group,
        block_ms=5000,
    )
    import asyncio  # local import keeps lifespan importable without asyncio at module load.

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
    app.state.graph = graph
    app.state.llm_caller = llm_caller

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
        await wecom_outbound.aclose()
        await feishu_outbound.aclose()
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
