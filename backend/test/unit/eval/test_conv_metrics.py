"""Unit tests for the conversational-eval feedback metrics."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.eval.common import TokenCounter
from backend.eval.conversational.cases import ConvResult
from backend.eval.conversational.judge import _Verdict, judge_task_completion


def _gen(prompt: int, completion: int) -> SimpleNamespace:
    return SimpleNamespace(
        generation_info={"token_usage": {"prompt_tokens": prompt, "completion_tokens": completion}}
    )


class TestTokenCounter:
    def test_tallies_across_calls(self) -> None:
        tc = TokenCounter()
        tc.on_llm_end(SimpleNamespace(generations=[[_gen(10, 5)]]))
        tc.on_llm_end(SimpleNamespace(generations=[[_gen(20, 8)]]))
        assert tc.prompt_tokens == 30
        assert tc.completion_tokens == 13
        assert tc.total_tokens == 43

    def test_missing_usage_is_zero(self) -> None:
        tc = TokenCounter()
        tc.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(generation_info=None)]]))
        assert tc.total_tokens == 0


class TestConvResultMetrics:
    def test_tool_success_rate(self) -> None:
        r = ConvResult(
            case_id="C1",
            persona="p",
            category="手机数码",
            gold_source_ids=["P1"],
            retrieved_source_ids=["P1"],
            hit=True,
            hit_turn=1,
            total_turns=2,
            tool_calls=4,
            tool_errors=1,
        )
        assert r.tool_success_rate == 0.75

    def test_tool_success_rate_no_calls_is_one(self) -> None:
        r = ConvResult(
            case_id="C1",
            persona="p",
            category="c",
            gold_source_ids=[],
            retrieved_source_ids=[],
            hit=False,
            hit_turn=0,
            total_turns=1,
            tool_calls=0,
            tool_errors=0,
        )
        assert r.tool_success_rate == 1.0

    def test_to_dict_includes_feedback_fields(self) -> None:
        r = ConvResult(
            case_id="C1",
            persona="p",
            category="c",
            gold_source_ids=[],
            retrieved_source_ids=[],
            hit=False,
            hit_turn=0,
            total_turns=3,
            task_completed=True,
            task_reason="解决了",
            tool_calls=2,
            tool_errors=0,
            prompt_tokens=100,
            completion_tokens=40,
            latency_ms=1234.5,
        )
        d = r.to_dict()
        assert d["task_completed"] is True
        assert d["task_reason"] == "解决了"
        assert d["total_tokens"] == 140
        assert d["latency_ms"] == 1234.5


def _judge_llm(completed: bool, reason: str) -> AsyncMock:
    llm = AsyncMock()
    llm.chat = AsyncMock(
        return_value=SimpleNamespace(parsed=_Verdict(completed=completed, reason=reason))
    )
    return llm


def _case(need: str = "想买两三千的安卓手机"):
    return SimpleNamespace(case_id="C1", hidden_need=need)


class TestJudge:
    async def test_completed_verdict(self) -> None:
        llm = _judge_llm(True, "推荐了符合预算的机型")
        ok, reason = await judge_task_completion(
            llm,
            case=_case(),
            transcript=[
                {"role": "user", "content": "推荐手机"},
                {"role": "agent", "content": "..."},
            ],
        )
        assert ok is True
        assert "符合预算" in reason

    async def test_empty_transcript_is_incomplete(self) -> None:
        llm = _judge_llm(True, "x")
        ok, _reason = await judge_task_completion(llm, case=_case(), transcript=[])
        assert ok is False
        llm.chat.assert_not_awaited()  # short-circuits, no LLM call

    async def test_llm_failure_degrades(self) -> None:
        llm = AsyncMock()
        llm.chat = AsyncMock(side_effect=RuntimeError("api down"))
        ok, reason = await judge_task_completion(
            llm, case=_case(), transcript=[{"role": "user", "content": "x"}]
        )
        assert ok is False
        assert "judge_error" in reason
