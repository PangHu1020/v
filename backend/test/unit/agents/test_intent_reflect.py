"""Unit tests for :mod:`backend.v.agents.intent_reflect`."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.v.agents.intent_reflect import (
    MAX_REFLECTION_RETRIES,
    IntentResult,
    ReflectionResult,
    intent_node,
    reflection_node,
)
from backend.v.models.llm_caller import LLMResult


def _llm(payload: dict[str, Any], *, model_cls: type | None = None) -> AsyncMock:
    if model_cls is None:
        model_cls = IntentResult if "intent" in payload else ReflectionResult
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
        caller = _llm({"intent": "refund", "confidence": 0.9})
        state = {"messages": [HumanMessage(content="我要退款")]}
        out = await intent_node(state, {"configurable": {"llm_caller": caller}})
        assert out["intent"] == "refund"

    async def test_no_caller_returns_general(self) -> None:
        out = await intent_node({"messages": [HumanMessage(content="hi")]}, {"configurable": {}})
        assert out == {"intent": "general"}

    async def test_no_human_message_returns_general(self) -> None:
        caller = _llm({"intent": "refund"})
        state = {"messages": [AIMessage(content="只有 AI")]}
        out = await intent_node(state, {"configurable": {"llm_caller": caller}})
        assert out == {"intent": "general"}
        caller.chat.assert_not_called()

    async def test_llm_failure_falls_back(self) -> None:
        caller = AsyncMock()
        caller.chat = AsyncMock(side_effect=RuntimeError("boom"))
        out = await intent_node(
            {"messages": [HumanMessage(content="退货")]},
            {"configurable": {"llm_caller": caller}},
        )
        assert out == {"intent": "general"}


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
