"""Unit tests for ``backend.v.agents.graph``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.v.agents.checkpoints.redis import RedisCheckpointer
from backend.v.agents.graph import build_graph
from backend.v.models.llm_caller import LLMResult


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


@pytest.fixture
def llm_caller() -> AsyncMock:
    caller = AsyncMock()
    caller.chat = AsyncMock(
        return_value=LLMResult(
            message=AIMessage(content="reply"),
            model="deepseek-flash",
            role="main_primary",
            fallback_used=False,
            latency_ms=10,
        )
    )
    return caller


class TestGraphBuild:
    async def test_compiles_and_runs_single_turn(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
        llm_caller: AsyncMock,
    ) -> None:
        ckpt = RedisCheckpointer(redis_client, ttl_seconds=600)
        graph = build_graph(ckpt)

        config = {
            "configurable": {
                "thread_id": "thread-1",
                "llm_caller": llm_caller,
            }
        }
        final = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="hello")],
                "session_id": "s-1",
                "channel": "wecom",
                "channel_user_id": "ext-1",
                "user_profile": {"member_level": "黄金"},
            },
            config=config,
        )

        msgs = final["messages"]
        # System (from enter_node) + Human (input) + AI (from agent_node).
        assert any(isinstance(m, SystemMessage) for m in msgs)
        assert any(isinstance(m, HumanMessage) and m.content == "hello" for m in msgs)
        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1
        assert ai_msgs[0].content == "reply"

    async def test_continuation_does_not_double_system_prompt(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
        llm_caller: AsyncMock,
    ) -> None:
        ckpt = RedisCheckpointer(redis_client, ttl_seconds=600)
        graph = build_graph(ckpt)

        config = {"configurable": {"thread_id": "thread-2", "llm_caller": llm_caller}}
        await graph.ainvoke(
            {
                "messages": [HumanMessage(content="turn 1")],
                "session_id": "s-2",
                "channel": "feishu",
                "channel_user_id": "ou_1",
                "user_profile": {},
            },
            config=config,
        )
        final = await graph.ainvoke(
            {"messages": [HumanMessage(content="turn 2")]},
            config=config,
        )
        sys_count = sum(1 for m in final["messages"] if isinstance(m, SystemMessage))
        assert sys_count == 1
