"""Phase-1 acceptance test: customer message in -> AI reply out.

Mounts the WeCom router on a FastAPI app, wires it to a real Debouncer
+ BusProducer + BusConsumer + LangGraph + RedisCheckpointer, with mocks
only at two boundaries: the LLM call (``LLMCaller.chat``) and the channel
outbound (``send_to``).

Verifies:
1. POST encrypted WeCom webhook returns 200 OK with body ``success``
   immediately (the work is async).
2. Within ~2 seconds, the mocked outbound has been called with the
   expected ``channel_user_id`` and the canned LLM reply.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import fakeredis.aioredis
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

from backend.app.bus.consumer import BusConsumer
from backend.app.bus.producer import BusProducer
from backend.app.bus.shard import RedisStreamShard
from backend.app.bus.worker import make_bus_handler
from backend.app.channels.debounce import Debouncer
from backend.app.channels.wecom.crypto import WecomCrypto
from backend.app.channels.wecom.router import build_router as build_wecom_router
from backend.app.channels.wecom.signature import compute_signature
from backend.app.gateway.middleware import RequestIdMiddleware
from backend.v.agents.graph import build_graph
from backend.v.configs.base import WecomSettings
from backend.v.memory.checkpointer import RedisCheckpointer
from backend.v.models.llm_caller import LLMResult

TEST_AES_KEY = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ"
TEST_CORP_ID = "wxabcdef1234567890"
TEST_TOKEN = "qwerty"
TEST_AGENT_ID = "1000002"

pytestmark = pytest.mark.e2e


def _fake_pool() -> MagicMock:
    pool = MagicMock(spec=asyncpg.Pool)
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=None)

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    return pool


@pytest.fixture
async def stack() -> AsyncIterator[dict]:
    """Build the full inbound stack and capture outbound calls."""
    redis = fakeredis.aioredis.FakeRedis(decode_responses=False)

    shard = RedisStreamShard(redis, prefix="bus", shard_count=2)
    producer = BusProducer(shard)
    consumer = BusConsumer(shard, group="g", consumer_name="c1", block_ms=20)

    crypto = WecomCrypto(TEST_AES_KEY, TEST_CORP_ID)

    async def channel_to_bus(msg) -> None:
        await producer.enqueue(msg)

    debouncer = Debouncer(redis, window_ms=80, dispatch=channel_to_bus)

    settings = WecomSettings(
        _env_file=None,  # type: ignore[call-arg]
        corp_id=TEST_CORP_ID,
        agent_id=TEST_AGENT_ID,
        secret="s",
        token=TEST_TOKEN,
        aes_key=TEST_AES_KEY,
    )

    ckpt = RedisCheckpointer(redis, ttl_seconds=600)
    graph = build_graph(ckpt)

    llm_caller = MagicMock()
    llm_caller.chat = AsyncMock(
        return_value=LLMResult(
            message=AIMessage(content="您好，已收到您的咨询，正在为您处理。"),
            model="deepseek-flash",
            role="main_primary",
            fallback_used=False,
            latency_ms=12,
        )
    )

    sent: list[tuple[str, str]] = []

    async def wecom_send(uid: str, text: str) -> None:
        sent.append((uid, text))

    handler = make_bus_handler(
        graph=graph,
        pool=_fake_pool(),
        redis=redis,
        llm_caller=llm_caller,
        sends={"wecom": wecom_send},
        silence_seconds=1800,
        cache_ttl_seconds=120,
    )

    consumer_task = asyncio.create_task(consumer.run(handler))

    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    app.include_router(build_wecom_router(settings, crypto, debouncer))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        yield {
            "client": client,
            "crypto": crypto,
            "sent": sent,
            "llm_caller": llm_caller,
        }

    await consumer.stop()
    await debouncer.shutdown()
    consumer_task.cancel()
    try:
        await consumer_task
    except (asyncio.CancelledError, BaseException):
        pass
    await redis.aclose()


def _wecom_event(crypto: WecomCrypto, *, from_user: str, content: str) -> tuple[str, dict]:
    """Build an encrypted WeCom POST body + matching signed query params."""
    inner = (
        "<xml>"
        "<ToUserName><![CDATA[corp]]></ToUserName>"
        f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
        "<CreateTime>1700000000</CreateTime>"
        "<MsgType><![CDATA[text]]></MsgType>"
        f"<Content><![CDATA[{content}]]></Content>"
        "<MsgId>1234567</MsgId>"
        f"<AgentID>{TEST_AGENT_ID}</AgentID>"
        "</xml>"
    )
    encrypted = crypto.encrypt(inner)
    envelope = (
        "<xml>"
        "<ToUserName><![CDATA[corp]]></ToUserName>"
        f"<Encrypt><![CDATA[{encrypted}]]></Encrypt>"
        "</xml>"
    )
    sig = compute_signature(TEST_TOKEN, "ts", "n", encrypted)
    return envelope, {
        "msg_signature": sig,
        "timestamp": "ts",
        "nonce": "n",
    }


async def _wait_for(predicate, *, timeout: float = 3.0, interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


class TestInboundToReply:
    async def test_full_round_trip(self, stack: dict) -> None:
        client = stack["client"]
        crypto = stack["crypto"]
        sent = stack["sent"]
        llm_caller = stack["llm_caller"]

        envelope, params = _wecom_event(
            crypto, from_user="ext-acceptance", content="订单 ORD123 状态？"
        )

        resp = await client.post("/webhook/wecom", params=params, content=envelope)
        assert resp.status_code == 200
        assert resp.text == "success"

        ok = await _wait_for(lambda: len(sent) >= 1, timeout=3.0)
        assert ok, "outbound send was never invoked"

        uid, reply = sent[0]
        assert uid == "ext-acceptance"
        assert reply == "您好，已收到您的咨询，正在为您处理。"

        # The LLMCaller.chat boundary was reached exactly once: one webhook,
        # one debounced message, one graph turn, one LLM call.
        assert llm_caller.chat.await_count == 1

    async def test_burst_collapses_into_one_reply(self, stack: dict) -> None:
        client = stack["client"]
        crypto = stack["crypto"]
        sent = stack["sent"]
        llm_caller = stack["llm_caller"]

        for content in ("你好", "我想问一下", "订单 ORD123 在哪"):
            envelope, params = _wecom_event(crypto, from_user="ext-burst", content=content)
            resp = await client.post("/webhook/wecom", params=params, content=envelope)
            assert resp.status_code == 200
            await asyncio.sleep(0.02)

        ok = await _wait_for(lambda: len(sent) >= 1, timeout=3.0)
        assert ok

        # Debouncer collapses three messages into a single agent turn.
        assert len(sent) == 1
        assert llm_caller.chat.await_count == 1
