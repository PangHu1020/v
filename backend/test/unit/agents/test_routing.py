"""Tests for the new routing functions and graph paths added in the
intent-routing + conditional-reflection refactor."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.v.agents.checkpoints.redis import RedisCheckpointer
from backend.v.agents.edges import route_after_agent, route_after_intent, route_after_reflect
from backend.v.agents.graph import build_graph
from backend.v.agents.nodes import agent_fast_node
from backend.v.models.llm_caller import LLMResult


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


@pytest.fixture
def mock_caller() -> AsyncMock:
    caller = AsyncMock()
    caller.chat = AsyncMock(
        return_value=LLMResult(
            message=AIMessage(content="reply"),
            model="m",
            role="main_primary",
            fallback_used=False,
            latency_ms=1,
        )
    )
    return caller


# ── route_after_intent ────────────────────────────────────────────────────────


class TestRouteAfterIntent:
    def test_general_routes_to_agent_fast(self) -> None:
        assert route_after_intent({"intent": "general"}) == "agent_fast"

    def test_refund_routes_to_agent(self) -> None:
        assert route_after_intent({"intent": "refund"}) == "agent"

    def test_logistics_routes_to_agent(self) -> None:
        assert route_after_intent({"intent": "logistics"}) == "agent"

    def test_complaint_routes_to_agent(self) -> None:
        assert route_after_intent({"intent": "complaint"}) == "agent"

    def test_missing_intent_routes_to_agent(self) -> None:
        assert route_after_intent({}) == "agent"


# ── route_after_agent ─────────────────────────────────────────────────────────


class TestRouteAfterAgent:
    def test_tool_calls_route_to_tools(self) -> None:
        ai = AIMessage(
            content="", tool_calls=[{"name": "search", "args": {}, "id": "1", "type": "tool_call"}]
        )
        assert route_after_agent({"messages": [ai]}) == "tools"

    def test_needs_reflection_routes_to_reflect(self) -> None:
        assert (
            route_after_agent({"messages": [AIMessage(content="x")], "needs_reflection": True})
            == "reflect"
        )

    def test_no_reflection_needed_exits_directly(self) -> None:
        assert (
            route_after_agent({"messages": [AIMessage(content="x")], "needs_reflection": False})
            == "exit"
        )

    def test_empty_messages_exits(self) -> None:
        assert route_after_agent({"messages": []}) == "exit"


# ── route_after_reflect ───────────────────────────────────────────────────────


class TestRouteAfterReflect:
    def test_reflection_failed_retries_agent(self) -> None:
        assert route_after_reflect({"reflection_failed": True}) == "agent"

    def test_reflection_passed_exits(self) -> None:
        assert route_after_reflect({"reflection_failed": False}) == "exit"

    def test_missing_flag_exits(self) -> None:
        assert route_after_reflect({}) == "exit"


# ── agent_fast_node ───────────────────────────────────────────────────────────


class TestAgentFastNode:
    async def test_calls_llm_without_tools(self, mock_caller: AsyncMock) -> None:
        result = await agent_fast_node(
            {"messages": [HumanMessage(content="hello")]},
            {"configurable": {"llm_caller": mock_caller}},
        )
        assert isinstance(result["messages"][0], AIMessage)
        # Must be called with tools=[] (no tool schema sent to LLM)
        call_kwargs = mock_caller.chat.call_args
        assert call_kwargs.kwargs.get("tools") == [] or call_kwargs.args[2:] == ()


# ── end-to-end: needs_reflection field propagates ────────────────────────────


class TestIntentReflectionPropagation:
    async def test_logistics_intent_sets_needs_reflection(
        self, redis_client: fakeredis.aioredis.FakeRedis, mock_caller: AsyncMock
    ) -> None:
        """When intent_node classifies logistics, needs_reflection should be True."""
        from backend.v.agents.intent_reflect import IntentResult

        intent_result = LLMResult(
            message=AIMessage(content=""),
            model="m",
            role="summary",
            fallback_used=False,
            latency_ms=1,
            parsed=IntentResult(intent="logistics", confidence=0.9),
        )

        # First call returns intent result, subsequent return normal reply
        mock_caller.chat = AsyncMock(side_effect=[intent_result, mock_caller.chat.return_value])

        ckpt = RedisCheckpointer(redis_client, ttl_seconds=600)
        graph = build_graph(ckpt)
        final = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="我的快递到哪了")],
                "channel": "wecom",
                "channel_user_id": "u1",
            },
            config={"configurable": {"thread_id": "t-logistics", "llm_caller": mock_caller}},
        )
        assert final.get("intent") == "logistics"
        assert final.get("needs_reflection") is True
