"""Tests for the V2 intent-routing functions, derivation logic, and graph paths."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.v.agents.checkpoints.redis import RedisCheckpointer
from backend.v.agents.edges import route_after_agent, route_after_intent, route_after_reflect
from backend.v.agents.graph import build_graph
from backend.v.agents.intent_reflect import IntentScores, _derive_routing
from backend.v.agents.nodes import agent_fast_node
from backend.v.models.llm_caller import LLMResult

_THRESH = {
    "refund": 0.6,
    "logistics": 0.55,
    "complaint": 0.6,
    "general": 0.45,
    "chitchat": 0.5,
}


def _derive(scores: dict, *, clarify_count: int = 0, max_clarify: int = 1, margin: float = 0.15):
    return _derive_routing(
        scores,
        thresholds=_THRESH,
        margin=margin,
        clarify_count=clarify_count,
        max_clarify=max_clarify,
    )


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


def _scores(**kw) -> dict[str, float]:
    base = {"refund": 0.0, "logistics": 0.0, "complaint": 0.0, "general": 0.0, "chitchat": 0.0}
    base.update(kw)
    return base


# ── _derive_routing: the core distribution → state logic ──────────────────────


class TestDeriveRouting:
    def test_single_clear_intent(self) -> None:
        out = _derive(_scores(refund=0.9, general=0.05))
        assert out["intent"] == "refund"
        assert out["is_mix"] is False
        assert out["needs_clarify"] is False
        assert out["needs_reflection"] is True  # refund is high-stakes

    def test_chitchat_single_no_reflection(self) -> None:
        out = _derive(_scores(chitchat=0.95, general=0.03))
        assert out["intent"] == "chitchat"
        assert out["needs_reflection"] is False

    def test_mix_when_top2_both_clear(self) -> None:
        # refund 0.7 ≥ 0.6 and logistics 0.65 ≥ 0.55 → genuine multi-intent.
        out = _derive(_scores(refund=0.7, logistics=0.65))
        assert out["is_mix"] is True
        assert out["intent"] == "refund"
        assert out["secondary_intent"] == "logistics"
        assert out["needs_reflection"] is True  # mix always reflects
        assert out["needs_clarify"] is False

    def test_ambiguous_top1_below_threshold(self) -> None:
        # Highest is refund 0.5 < 0.6 → classifier unsure → clarify.
        out = _derive(_scores(refund=0.5, complaint=0.3))
        assert out["needs_clarify"] is True
        assert out["is_mix"] is False

    def test_ambiguous_close_margin(self) -> None:
        # general 0.5 ≥ 0.45 clears, but logistics 0.42 is within 0.15 margin
        # and does NOT clear its 0.55 bar → not mix, but too close → clarify.
        out = _derive(_scores(general=0.5, logistics=0.42))
        assert out["needs_clarify"] is True
        assert out["is_mix"] is False

    def test_clarify_capped_falls_back_to_top1(self) -> None:
        # Same ambiguous distribution, but the cap is already reached.
        out = _derive(_scores(refund=0.5, complaint=0.3), clarify_count=1, max_clarify=1)
        assert out["needs_clarify"] is False
        assert out["intent"] == "refund"  # fall back to top-1

    def test_single_clear_with_wide_margin_not_ambiguous(self) -> None:
        out = _derive(_scores(logistics=0.8, general=0.1))
        assert out["needs_clarify"] is False
        assert out["is_mix"] is False
        assert out["intent"] == "logistics"


# ── route_after_intent ────────────────────────────────────────────────────────


class TestRouteAfterIntent:
    def test_needs_clarify_routes_to_clarify(self) -> None:
        assert route_after_intent({"needs_clarify": True, "intent": "refund"}) == "clarify"

    def test_chitchat_single_routes_to_agent_fast(self) -> None:
        assert route_after_intent({"intent": "chitchat", "is_mix": False}) == "agent_fast"

    def test_chitchat_mix_routes_to_agent(self) -> None:
        # A mix that happens to have chitchat on top still needs the full agent.
        assert route_after_intent({"intent": "chitchat", "is_mix": True}) == "agent"

    def test_refund_routes_to_agent(self) -> None:
        assert route_after_intent({"intent": "refund"}) == "agent"

    def test_general_routes_to_agent(self) -> None:
        # general now goes to the full agent (may need tools), not fast.
        assert route_after_intent({"intent": "general"}) == "agent"

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
        call_kwargs = mock_caller.chat.call_args
        assert call_kwargs.kwargs.get("tools") == [] or call_kwargs.args[2:] == ()


# ── end-to-end: distribution drives routing + reflection ─────────────────────


class TestIntentRoutingE2E:
    async def test_mix_intent_sets_reflection(
        self, redis_client: fakeredis.aioredis.FakeRedis, mock_caller: AsyncMock
    ) -> None:
        """A mix distribution (refund+logistics) sets needs_reflection + is_mix."""
        intent_result = LLMResult(
            message=AIMessage(content=""),
            model="m",
            role="summary",
            fallback_used=False,
            latency_ms=1,
            parsed=IntentScores(refund=0.7, logistics=0.65),
        )
        mock_caller.chat = AsyncMock(side_effect=[intent_result, mock_caller.chat.return_value])

        ckpt = RedisCheckpointer(redis_client, ttl_seconds=600)
        graph = build_graph(ckpt)
        final = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="我要退款，顺便问下快递到哪了")],
                "channel": "wecom",
                "channel_user_id": "u1",
            },
            config={"configurable": {"thread_id": "t-mix", "llm_caller": mock_caller}},
        )
        assert final.get("intent") == "refund"
        assert final.get("is_mix") is True
        assert final.get("needs_reflection") is True

    async def test_ambiguous_routes_through_clarify(
        self, redis_client: fakeredis.aioredis.FakeRedis, mock_caller: AsyncMock
    ) -> None:
        """An ambiguous distribution goes through clarify → agent, with the
        clarify directive injected into the agent's LLM call (not persisted)."""
        intent_result = LLMResult(
            message=AIMessage(content=""),
            model="m",
            role="summary",
            fallback_used=False,
            latency_ms=1,
            parsed=IntentScores(refund=0.5, complaint=0.3),
        )
        captured: dict = {}

        async def _chat(role, messages, **kw):
            if role == "summary":
                return intent_result
            captured["messages"] = messages
            return LLMResult(
                message=AIMessage(content="reply"),
                model="m",
                role="main_primary",
                fallback_used=False,
                latency_ms=1,
            )

        mock_caller.chat = AsyncMock(side_effect=_chat)

        ckpt = RedisCheckpointer(redis_client, ttl_seconds=600)
        graph = build_graph(ckpt)
        final = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="这个东西有点问题")],
                "channel": "wecom",
                "channel_user_id": "u1",
            },
            config={"configurable": {"thread_id": "t-amb", "llm_caller": mock_caller}},
        )
        assert final.get("clarify_count") == 1
        # Directive was injected for the agent call but not written to history.
        agent_msgs = captured.get("messages", [])
        assert any("意图不明确" in getattr(m, "content", "") for m in agent_msgs)
        assert all("意图不明确" not in getattr(m, "content", "") for m in final["messages"])
        # Invariant: the directive is appended at the TAIL (append-only, cache-stable
        # prefix) and is NOT a SystemMessage — only the frozen head carries system
        # messages. A mid-stream SystemMessage breaks strict templates (vllm Qwen).
        from langchain_core.messages import SystemMessage

        directive_msg = next(m for m in agent_msgs if "意图不明确" in getattr(m, "content", ""))
        assert not isinstance(directive_msg, SystemMessage)
        assert agent_msgs[-1] is directive_msg  # appended last, nothing after it
        # No SystemMessage may follow a non-system message anywhere in the call.
        first_non_system = next(
            (i for i, m in enumerate(agent_msgs) if not isinstance(m, SystemMessage)),
            len(agent_msgs),
        )
        assert all(not isinstance(m, SystemMessage) for m in agent_msgs[first_non_system:])
