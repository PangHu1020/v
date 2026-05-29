"""Unit tests for ``backend.v.agents.intent_reflect`` (Phase-3 Groups E + F).

Covers:

- :func:`intent_node` happy path (LLM returns valid IntentResult JSON).
- :func:`intent_node` falls back to ``"general"`` when:
    - no LLM caller is configured,
    - no human message is in state,
    - the LLM raises.
- :func:`intent_node` emotion pre-emption: when an
  :class:`EmotionDetector` is supplied via config and the latest message
  scores above threshold, the node returns
  ``{"intent": "complaint", "force_handoff": True}`` without calling the
  LLM.
- :func:`reflection_node` passes / fails / max-retries paths.
- :func:`route_after_reflect` if exposed.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.v.agents.intent_reflect import (
    MAX_REFLECTION_RETRIES,
    intent_node,
    reflection_node,
)
from backend.v.models.llm_caller import LLMResult


def _llm(payload: dict[str, Any]) -> AsyncMock:
    caller = AsyncMock()
    caller.chat = AsyncMock(
        return_value=LLMResult(
            message=AIMessage(content=json.dumps(payload)),
            model="deepseek-flash",
            role="summary",
            fallback_used=False,
            latency_ms=5,
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
        state = {"messages": [HumanMessage(content="hi")]}
        out = await intent_node(state, {"configurable": {}})
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
        state = {"messages": [HumanMessage(content="退货")]}
        out = await intent_node(state, {"configurable": {"llm_caller": caller}})
        assert out == {"intent": "general"}

    async def test_emotion_preempts_handoff(self) -> None:
        caller = _llm({"intent": "refund"})
        detector = AsyncMock()
        detector.score = AsyncMock(return_value=0.95)
        state = {"messages": [HumanMessage(content="草你妈")]}
        out = await intent_node(
            state,
            {
                "configurable": {
                    "llm_caller": caller,
                    "emotion_detector": detector,
                    "emotion_threshold": 0.80,
                }
            },
        )
        assert out == {"intent": "complaint", "force_handoff": True}
        # The LLM is NOT called when pre-emption fires.
        caller.chat.assert_not_called()

    async def test_emotion_below_threshold_continues(self) -> None:
        caller = _llm({"intent": "general"})
        detector = AsyncMock()
        detector.score = AsyncMock(return_value=0.30)
        state = {"messages": [HumanMessage(content="请问怎么退款")]}
        out = await intent_node(
            state,
            {
                "configurable": {
                    "llm_caller": caller,
                    "emotion_detector": detector,
                    "emotion_threshold": 0.80,
                }
            },
        )
        assert out == {"intent": "general"}
        caller.chat.assert_called_once()

    async def test_emotion_detector_failure_does_not_block(self) -> None:
        # Detector raising should be swallowed; LLM still runs.
        caller = _llm({"intent": "general"})
        detector = AsyncMock()
        detector.score = AsyncMock(side_effect=RuntimeError("score boom"))
        state = {"messages": [HumanMessage(content="hi")]}
        out = await intent_node(
            state,
            {
                "configurable": {
                    "llm_caller": caller,
                    "emotion_detector": detector,
                    "emotion_threshold": 0.80,
                }
            },
        )
        assert out["intent"] == "general"


class TestReflectionNode:
    async def test_passes_returns_empty(self) -> None:
        caller = _llm({"passes": True, "issues": []})
        state = {
            "messages": [
                HumanMessage(content="物流"),
                AIMessage(content="您的快递已签收"),
            ]
        }
        out = await reflection_node(state, {"configurable": {"llm_caller": caller}})
        assert out == {}

    async def test_fails_increments_retries(self) -> None:
        caller = _llm({"passes": False, "issues": ["编造单号"]})
        state = {
            "messages": [
                HumanMessage(content="物流"),
                AIMessage(content="您的快递单号是 SF999"),
            ]
        }
        out = await reflection_node(state, {"configurable": {"llm_caller": caller}})
        assert out["reflection_failed"] is True
        assert out["reflection_retries"] == 1

    async def test_max_retries_short_circuits(self) -> None:
        caller = _llm({"passes": False})
        state = {
            "messages": [AIMessage(content="x")],
            "reflection_retries": MAX_REFLECTION_RETRIES,
        }
        out = await reflection_node(state, {"configurable": {"llm_caller": caller}})
        assert out == {}
        caller.chat.assert_not_called()

    async def test_no_caller_returns_empty(self) -> None:
        state = {"messages": [AIMessage(content="x")]}
        out = await reflection_node(state, {"configurable": {}})
        assert out == {}

    async def test_no_ai_reply_returns_empty(self) -> None:
        caller = _llm({"passes": True})
        state = {"messages": [HumanMessage(content="hi")]}
        out = await reflection_node(state, {"configurable": {"llm_caller": caller}})
        assert out == {}

    async def test_collects_tool_messages_into_facts(self) -> None:
        caller = _llm({"passes": True})
        state = {
            "messages": [
                HumanMessage(content="物流"),
                ToolMessage(content="单号 SF999", tool_call_id="tc1"),
                AIMessage(content="您的快递单号 SF999 已签收"),
            ]
        }
        await reflection_node(state, {"configurable": {"llm_caller": caller}})
        prompt_arg = caller.chat.call_args.args[1]
        # The reflection prompt should embed the tool message content.
        joined = "\n".join(m.content for m in prompt_arg if isinstance(m.content, str))
        assert "SF999" in joined

    async def test_llm_failure_returns_empty(self) -> None:
        caller = AsyncMock()
        caller.chat = AsyncMock(side_effect=RuntimeError("boom"))
        state = {
            "messages": [
                HumanMessage(content="物流"),
                AIMessage(content="您的快递已签收"),
            ]
        }
        out = await reflection_node(state, {"configurable": {"llm_caller": caller}})
        assert out == {}


class TestEmotionPreemptIntegration:
    """Pre-emption end-to-end: detector + threshold + state output shape."""

    @pytest.mark.parametrize("score,expected_force", [(0.95, True), (0.50, False)])
    async def test_threshold_boundary(self, score: float, expected_force: bool) -> None:
        caller = _llm({"intent": "general"})
        detector = AsyncMock()
        detector.score = AsyncMock(return_value=score)
        state = {"messages": [HumanMessage(content="x")]}
        out = await intent_node(
            state,
            {
                "configurable": {
                    "llm_caller": caller,
                    "emotion_detector": detector,
                    "emotion_threshold": 0.80,
                }
            },
        )
        assert out.get("force_handoff", False) is expected_force
