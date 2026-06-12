"""Unit tests for :mod:`backend.v.agents.intent_reflect`."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.v.agents.intent_reflect import (
    MAX_REFLECTION_RETRIES,
    IntentScores,
    ReflectionResult,
    clarify_node,
    intent_node,
    reflection_node,
)
from backend.v.models.llm_caller import LLMResult


def _intent_llm(scores: dict[str, float]) -> AsyncMock:
    parsed = IntentScores.model_validate(scores)
    caller = AsyncMock()
    caller.chat = AsyncMock(
        return_value=LLMResult(
            message=AIMessage(content=json.dumps(scores)),
            model="deepseek-flash",
            role="summary",
            fallback_used=False,
            latency_ms=5,
            parsed=parsed,
        )
    )
    return caller


def _llm(payload: dict[str, Any], *, model_cls: type | None = None) -> AsyncMock:
    # Reflection-only helper (intent now uses _intent_llm with a distribution).
    if model_cls is None:
        model_cls = ReflectionResult
    parsed = model_cls.model_validate(payload)
    caller = AsyncMock()
    caller.chat = AsyncMock(
        return_value=LLMResult(
            message=AIMessage(content=json.dumps(payload)),
            model="deepseek-flash",
            role="summary",
            fallback_used=False,
            latency_ms=5,
            parsed=parsed,
        )
    )
    return caller


class TestIntentNode:
    async def test_classifies_refund(self) -> None:
        caller = _intent_llm({"refund": 0.9, "general": 0.05})
        state = {"messages": [HumanMessage(content="我要退款")]}
        out = await intent_node(state, {"configurable": {"llm_caller": caller}})
        assert out["intent"] == "refund"
        assert out["is_mix"] is False
        assert out["intent_scores"]["refund"] == 0.9

    async def test_mix_distribution(self) -> None:
        caller = _intent_llm({"refund": 0.7, "logistics": 0.65})
        state = {"messages": [HumanMessage(content="退款并查物流")]}
        out = await intent_node(state, {"configurable": {"llm_caller": caller}})
        assert out["is_mix"] is True
        assert out["needs_reflection"] is True

    async def test_ambiguous_sets_clarify(self) -> None:
        caller = _intent_llm({"refund": 0.5, "complaint": 0.3})
        state = {"messages": [HumanMessage(content="这个有问题")]}
        out = await intent_node(state, {"configurable": {"llm_caller": caller}})
        assert out["needs_clarify"] is True

    async def test_no_caller_returns_general(self) -> None:
        out = await intent_node({"messages": [HumanMessage(content="hi")]}, {"configurable": {}})
        assert out["intent"] == "general"

    async def test_no_human_message_returns_general(self) -> None:
        caller = _intent_llm({"refund": 0.9})
        state = {"messages": [AIMessage(content="只有 AI")]}
        out = await intent_node(state, {"configurable": {"llm_caller": caller}})
        assert out["intent"] == "general"
        caller.chat.assert_not_called()

    async def test_llm_failure_falls_back(self) -> None:
        caller = AsyncMock()
        caller.chat = AsyncMock(side_effect=RuntimeError("boom"))
        out = await intent_node(
            {"messages": [HumanMessage(content="退货")]},
            {"configurable": {"llm_caller": caller}},
        )
        assert out["intent"] == "general"

    async def test_resets_stale_directive(self) -> None:
        caller = _intent_llm({"refund": 0.9})
        out = await intent_node(
            {"messages": [HumanMessage(content="退款")], "intent_directive": "stale"},
            {"configurable": {"llm_caller": caller}},
        )
        assert out["intent_directive"] == ""


class TestClarifyNode:
    async def test_sets_directive_and_increments(self) -> None:
        out = await clarify_node(
            {"intent": "refund", "secondary_intent": "complaint", "clarify_count": 0}
        )
        assert "refund" in out["intent_directive"]
        assert "退款售后" in out["intent_directive"]
        assert out["clarify_count"] == 1

    async def test_increments_existing_count(self) -> None:
        out = await clarify_node(
            {"intent": "logistics", "secondary_intent": "general", "clarify_count": 1}
        )
        assert out["clarify_count"] == 2


class TestReflectionNode:
    async def test_passes_returns_empty(self) -> None:
        caller = _llm({"passes": True, "issues": []})
        state = {"messages": [HumanMessage(content="物流"), AIMessage(content="已签收")]}
        assert await reflection_node(state, {"configurable": {"llm_caller": caller}}) == {}

    async def test_fails_increments_retries(self) -> None:
        caller = _llm({"passes": False, "issues": ["编造单号"]})
        state = {"messages": [HumanMessage(content="物流"), AIMessage(content="单号 SF999")]}
        out = await reflection_node(state, {"configurable": {"llm_caller": caller}})
        assert out["reflection_failed"] is True
        assert out["reflection_retries"] == 1

    async def test_max_retries_short_circuits(self) -> None:
        caller = _llm({"passes": False})
        state = {"messages": [AIMessage(content="x")], "reflection_retries": MAX_REFLECTION_RETRIES}
        assert await reflection_node(state, {"configurable": {"llm_caller": caller}}) == {}
        caller.chat.assert_not_called()

    async def test_no_caller_returns_empty(self) -> None:
        state = {"messages": [AIMessage(content="x")]}
        assert await reflection_node(state, {"configurable": {}}) == {}

    async def test_no_ai_reply_returns_empty(self) -> None:
        caller = _llm({"passes": True})
        assert (
            await reflection_node(
                {"messages": [HumanMessage(content="hi")]}, {"configurable": {"llm_caller": caller}}
            )
            == {}
        )

    async def test_collects_tool_messages_into_facts(self) -> None:
        caller = _llm({"passes": True})
        state = {
            "messages": [
                HumanMessage(content="物流"),
                ToolMessage(content="单号 SF999", tool_call_id="tc1"),
                AIMessage(content="单号 SF999 已签收"),
            ]
        }
        await reflection_node(state, {"configurable": {"llm_caller": caller}})
        joined = "\n".join(
            m.content for m in caller.chat.call_args.args[1] if isinstance(m.content, str)
        )
        assert "SF999" in joined

    async def test_llm_failure_returns_empty(self) -> None:
        caller = AsyncMock()
        caller.chat = AsyncMock(side_effect=RuntimeError("boom"))
        state = {"messages": [HumanMessage(content="物流"), AIMessage(content="已签收")]}
        assert await reflection_node(state, {"configurable": {"llm_caller": caller}}) == {}
