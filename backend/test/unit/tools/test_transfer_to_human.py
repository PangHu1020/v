"""Unit tests for ``backend.v.tools.transfer_to_human``.

The tool is not callable in isolation because :func:`langgraph.types.interrupt`
needs the LangGraph runtime. We exercise it via a tiny graph that:

1. Builds a single node which just runs the tool.
2. Hits the interrupt on first invoke.
3. Resumes with a structured ``decision`` payload.
4. Asserts the tool's return string contains the operator messages.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from backend.v.agents.checkpointer import RedisCheckpointer
from backend.v.tools import transfer_to_human


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


def _build_tool_graph(checkpointer):  # type: ignore[no-untyped-def]
    """A trivial graph that invokes the tool once and stores the result."""

    async def call_tool(state):
        result = await transfer_to_human.ainvoke({"reason": state["reason"]})
        return {"result": result}

    g = StateGraph(dict)
    g.add_node("call", call_tool)
    g.set_entry_point("call")
    g.add_edge("call", END)
    return g.compile(checkpointer=checkpointer)


class TestTransferToHumanTool:
    async def test_pauses_on_first_invoke(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        ckpt = RedisCheckpointer(redis_client, ttl_seconds=600)
        graph = _build_tool_graph(ckpt)
        config: RunnableConfig = {"configurable": {"thread_id": "t-1"}}

        # First invoke triggers interrupt; the result carries an
        # ``__interrupt__`` list and the user-facing keys are NOT yet
        # populated.
        result = await graph.ainvoke({"reason": "客户要求退款"}, config=config)
        assert "result" not in result

        interrupts = result.get("__interrupt__") or []
        assert len(interrupts) == 1
        value = interrupts[0].value
        assert value["type"] == "transfer_to_human"
        assert value["reason"] == "客户要求退款"

        # Snapshot also reports the graph as paused.
        snap = await graph.aget_state(config)
        assert snap.next, "expected the graph to be paused at an interrupt"

    async def test_resume_with_operator_messages(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        ckpt = RedisCheckpointer(redis_client, ttl_seconds=600)
        graph = _build_tool_graph(ckpt)
        config: RunnableConfig = {"configurable": {"thread_id": "t-2"}}

        await graph.ainvoke({"reason": "复杂投诉"}, config=config)

        decision = {
            "type": "resume",
            "operator_messages": ["请告知客户已升级到主管", "建议提供 5% 折扣"],
        }
        final = await graph.ainvoke(Command(resume=decision), config=config)
        assert "人工已接管" in final["result"]
        assert "已升级到主管" in final["result"]
        assert "5% 折扣" in final["result"]

    async def test_resume_with_empty_operator_messages(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        ckpt = RedisCheckpointer(redis_client, ttl_seconds=600)
        graph = _build_tool_graph(ckpt)
        config: RunnableConfig = {"configurable": {"thread_id": "t-3"}}

        await graph.ainvoke({"reason": "确认"}, config=config)
        decision = {"type": "resume", "operator_messages": []}
        final = await graph.ainvoke(Command(resume=decision), config=config)
        assert "AI 继续" in final["result"]

    async def test_resume_with_string_decision(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        ckpt = RedisCheckpointer(redis_client, ttl_seconds=600)
        graph = _build_tool_graph(ckpt)
        config: RunnableConfig = {"configurable": {"thread_id": "t-4"}}

        await graph.ainvoke({"reason": "简单确认"}, config=config)
        final = await graph.ainvoke(Command(resume="continue"), config=config)
        assert "continue" in final["result"]
