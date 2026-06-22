"""Unit tests for ``backend.v.hooks.tool_guard`` (Phase-3 Group D).

Covers the four pure helpers plus the two combined evaluators that the
guarded ToolNode uses inside the compiled graph:

- :func:`compute_fingerprint` — canonical ``tool:args`` key.
- :func:`check_loop` — count-and-stop semantics.
- :func:`is_circuit_open` — error-count threshold.
- :func:`is_tool_error` — heuristic on ToolMessage content.
- :func:`evaluate_tool_calls` — combined dead-loop + circuit evaluation.
- :func:`update_error_counts` — post-tool error tracking.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.v.hooks.tool_guard import (
    BLOCK_BUDGET,
    BLOCK_CIRCUIT,
    BLOCK_LOOP,
    CIRCUIT_OPEN_THRESHOLD,
    LOOP_STOP_THRESHOLD,
    LOOP_WARN_THRESHOLD,
    TURN_MAX_TOOL_CALLS,
    block_message,
    check_loop,
    compute_fingerprint,
    count_tool_calls_since_last_human,
    evaluate_tool_calls,
    exceeds_turn_budget,
    is_circuit_open,
    is_tool_error,
    message_is_error,
    update_error_counts,
)


def _ai_with_calls(n: int) -> AIMessage:
    """An AIMessage carrying ``n`` distinct tool calls."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "search", "args": {"query": f"q{i}"}, "id": f"c{i}", "type": "tool_call"}
            for i in range(n)
        ],
    )


class TestCountToolCallsSinceLastHuman:
    def test_empty(self) -> None:
        assert count_tool_calls_since_last_human([]) == 0

    def test_no_human_counts_all(self) -> None:
        msgs = [_ai_with_calls(2), ToolMessage(content="x", tool_call_id="c0"), _ai_with_calls(3)]
        assert count_tool_calls_since_last_human(msgs) == 5

    def test_resets_at_last_human(self) -> None:
        # Calls before the latest HumanMessage are a prior turn — not counted.
        msgs = [
            _ai_with_calls(4),  # prior turn
            HumanMessage(content="new turn"),
            _ai_with_calls(2),  # this turn
        ]
        assert count_tool_calls_since_last_human(msgs) == 2

    def test_ai_without_tool_calls_ignored(self) -> None:
        msgs = [HumanMessage(content="hi"), AIMessage(content="just text")]
        assert count_tool_calls_since_last_human(msgs) == 0

    def test_multiple_calls_per_ai_summed(self) -> None:
        msgs = [HumanMessage(content="hi"), _ai_with_calls(3), _ai_with_calls(2)]
        assert count_tool_calls_since_last_human(msgs) == 5

    def test_cap_constant_sane(self) -> None:
        assert TURN_MAX_TOOL_CALLS >= 1


class TestExceedsTurnBudget:
    def test_under_budget_ok(self) -> None:
        msgs = [HumanMessage(content="hi"), _ai_with_calls(2)]
        # 2 prior + 1 pending = 3 ≤ cap
        assert exceeds_turn_budget(msgs, 1) is False

    def test_exactly_at_cap_ok(self) -> None:
        msgs = [HumanMessage(content="hi"), _ai_with_calls(TURN_MAX_TOOL_CALLS - 1)]
        assert exceeds_turn_budget(msgs, 1) is False

    def test_over_cap_blocks(self) -> None:
        msgs = [HumanMessage(content="hi"), _ai_with_calls(TURN_MAX_TOOL_CALLS)]
        # cap prior + 1 pending > cap
        assert exceeds_turn_budget(msgs, 1) is True

    def test_prior_turn_not_counted(self) -> None:
        # A big prior turn before the latest human must not count against this turn.
        msgs = [_ai_with_calls(50), HumanMessage(content="new turn")]
        assert exceeds_turn_budget(msgs, 1) is False


class TestComputeFingerprint:
    def test_stable_for_same_args(self) -> None:
        a = compute_fingerprint("recall_memory", {"query": "退款", "k": 3})
        b = compute_fingerprint("recall_memory", {"k": 3, "query": "退款"})
        assert a == b

    def test_different_args_differ(self) -> None:
        a = compute_fingerprint("recall_memory", {"query": "退款"})
        b = compute_fingerprint("recall_memory", {"query": "物流"})
        assert a != b

    def test_different_tools_differ(self) -> None:
        a = compute_fingerprint("recall_memory", {"q": "x"})
        b = compute_fingerprint("subagent", {"q": "x"})
        assert a != b

    def test_unicode_preserved(self) -> None:
        fp = compute_fingerprint("t", {"q": "中文"})
        assert "中文" in fp


class TestCheckLoop:
    def test_first_call_zero_count(self) -> None:
        count, stop = check_loop([], "tool:{}")
        assert count == 0
        assert stop is False

    def test_warn_threshold_does_not_stop(self) -> None:
        fps = ["t:{}"] * LOOP_WARN_THRESHOLD
        count, stop = check_loop(fps, "t:{}")
        assert count == LOOP_WARN_THRESHOLD
        assert stop is False

    def test_stop_threshold_triggers(self) -> None:
        fps = ["t:{}"] * LOOP_STOP_THRESHOLD
        count, stop = check_loop(fps, "t:{}")
        assert count == LOOP_STOP_THRESHOLD
        assert stop is True

    def test_unrelated_calls_dont_count(self) -> None:
        fps = ["other:{}"] * 10
        count, stop = check_loop(fps, "t:{}")
        assert count == 0
        assert stop is False


class TestIsCircuitOpen:
    def test_below_threshold(self) -> None:
        assert (
            is_circuit_open({"recall_memory": CIRCUIT_OPEN_THRESHOLD - 1}, "recall_memory") is False
        )

    def test_at_threshold(self) -> None:
        assert is_circuit_open({"recall_memory": CIRCUIT_OPEN_THRESHOLD}, "recall_memory") is True

    def test_unknown_tool(self) -> None:
        assert is_circuit_open({}, "anything") is False


class TestIsToolError:
    def test_bracket_prefix(self) -> None:
        assert is_tool_error("[error] tool failed") is True

    def test_error_keyword(self) -> None:
        assert is_tool_error("Some Error happened") is True

    def test_normal_content(self) -> None:
        assert is_tool_error("ok, result is 42") is False

    def test_non_string(self) -> None:
        assert is_tool_error({"ok": True}) is False
        assert is_tool_error(None) is False


class TestEvaluateToolCalls:
    def test_first_call_no_reason(self) -> None:
        calls = [{"id": "c1", "name": "recall_memory", "args": {"q": "a"}}]
        new_fps, reasons = evaluate_tool_calls(calls, [], {})
        assert len(new_fps) == 1
        assert reasons == [None]

    def test_loop_stop_gives_loop_reason(self) -> None:
        fp = compute_fingerprint("recall_memory", {"q": "a"})
        existing = [fp] * LOOP_STOP_THRESHOLD
        calls = [{"id": "c1", "name": "recall_memory", "args": {"q": "a"}}]
        new_fps, reasons = evaluate_tool_calls(calls, existing, {})
        assert new_fps == [fp]
        assert reasons[0]["reason"] == BLOCK_LOOP
        assert reasons[0]["tool"] == "recall_memory"
        assert reasons[0]["count"] == LOOP_STOP_THRESHOLD + 1

    def test_warn_does_not_block(self) -> None:
        fp = compute_fingerprint("recall_memory", {"q": "a"})
        existing = [fp] * LOOP_WARN_THRESHOLD
        calls = [{"id": "c1", "name": "recall_memory", "args": {"q": "a"}}]
        _, reasons = evaluate_tool_calls(calls, existing, {})
        assert reasons == [None]

    def test_circuit_open_gives_circuit_reason(self) -> None:
        calls = [{"id": "c1", "name": "recall_memory", "args": {"q": "a"}}]
        _, reasons = evaluate_tool_calls(
            calls,
            [],
            {"recall_memory": CIRCUIT_OPEN_THRESHOLD},
        )
        assert reasons[0]["reason"] == BLOCK_CIRCUIT
        assert reasons[0]["tool"] == "recall_memory"

    def test_circuit_precedence_over_loop(self) -> None:
        # A tool that is both looping AND circuit-open reports circuit (tool unhealthy).
        fp = compute_fingerprint("recall_memory", {"q": "a"})
        existing = [fp] * LOOP_STOP_THRESHOLD
        calls = [{"id": "c1", "name": "recall_memory", "args": {"q": "a"}}]
        _, reasons = evaluate_tool_calls(calls, existing, {"recall_memory": CIRCUIT_OPEN_THRESHOLD})
        assert reasons[0]["reason"] == BLOCK_CIRCUIT

    def test_only_offending_call_blocked(self) -> None:
        # The clean call stays allowed (None); only the looping one gets a reason.
        good = {"id": "c1", "name": "recall_memory", "args": {"q": "ok"}}
        bad = {"id": "c2", "name": "recall_memory", "args": {"q": "loop"}}
        loop_fp = compute_fingerprint("recall_memory", {"q": "loop"})
        existing = [loop_fp] * LOOP_STOP_THRESHOLD
        new_fps, reasons = evaluate_tool_calls([good, bad], existing, {})
        assert len(new_fps) == 2
        assert reasons[0] is None
        assert reasons[1]["reason"] == BLOCK_LOOP


class TestBlockMessage:
    def test_loop_mentions_tool_and_count(self) -> None:
        msg = block_message(BLOCK_LOOP, tool="search", count=5)
        assert msg.startswith("[tool_guard]")
        assert "search" in msg and "5" in msg

    def test_circuit_mentions_tool(self) -> None:
        msg = block_message(BLOCK_CIRCUIT, tool="search", count=3)
        assert "[tool_guard]" in msg and "search" in msg

    def test_budget_mentions_cap(self) -> None:
        msg = block_message(BLOCK_BUDGET, cap=8)
        assert "[tool_guard]" in msg and "8" in msg

    def test_unknown_reason_safe_fallback(self) -> None:
        msg = block_message("???")
        assert msg.startswith("[tool_guard]")


class TestUpdateErrorCounts:
    def test_increments_on_error_message(self) -> None:
        calls = [{"id": "c1", "name": "recall_memory", "args": {}}]
        msgs = [ToolMessage(content="[error] boom", tool_call_id="c1")]
        out = update_error_counts(calls, msgs, {})
        assert out["recall_memory"] == 1

    def test_does_not_increment_on_success(self) -> None:
        calls = [{"id": "c1", "name": "recall_memory", "args": {}}]
        msgs = [ToolMessage(content="ok", tool_call_id="c1")]
        out = update_error_counts(calls, msgs, {})
        assert out == {}

    def test_accumulates(self) -> None:
        calls = [{"id": "c1", "name": "recall_memory", "args": {}}]
        msgs = [ToolMessage(content="[error] boom", tool_call_id="c1")]
        out = update_error_counts(calls, msgs, {"recall_memory": 2})
        assert out["recall_memory"] == 3

    def test_unknown_tool_call_id_ignored(self) -> None:
        calls = [{"id": "c1", "name": "recall_memory", "args": {}}]
        msgs = [ToolMessage(content="[error]", tool_call_id="unknown")]
        out = update_error_counts(calls, msgs, {})
        assert out == {}

    def test_does_not_mutate_input(self) -> None:
        original = {"recall_memory": 1}
        calls = [{"id": "c1", "name": "recall_memory", "args": {}}]
        msgs = [ToolMessage(content="[error]", tool_call_id="c1")]
        update_error_counts(calls, msgs, original)
        assert original == {"recall_memory": 1}

    def test_status_error_counts_regardless_of_content(self) -> None:
        # A Chinese error string the content heuristic would miss, but
        # status="error" (set by ToolNode on a raise) catches it.
        calls = [{"id": "c1", "name": "search", "args": {}}]
        msgs = [ToolMessage(content="检索失败", tool_call_id="c1", status="error")]
        out = update_error_counts(calls, msgs, {})
        assert out["search"] == 1

    def test_empty_result_success_not_counted(self) -> None:
        # Clean empty-result (no error marker) with default success status → not an error.
        calls = [{"id": "c1", "name": "recall_memory", "args": {}}]
        msgs = [ToolMessage(content="（暂无相关历史记忆）", tool_call_id="c1")]
        out = update_error_counts(calls, msgs, {})
        assert out == {}

    def test_returned_error_marker_still_counted(self) -> None:
        # MCP/subagent signal failure by RETURNING "[tool_error]" → status stays
        # "success" (no raise), so the content fallback must still catch it.
        calls = [{"id": "c1", "name": "shop__query_order", "args": {}}]
        msgs = [ToolMessage(content="[tool_error] upstream 500", tool_call_id="c1")]
        out = update_error_counts(calls, msgs, {})
        assert out["shop__query_order"] == 1


class TestMessageIsError:
    def test_status_error_wins(self) -> None:
        # status="error" forces error even when content looks clean.
        assert message_is_error(ToolMessage(content="ok", tool_call_id="c", status="error"))

    def test_returned_error_marker_via_content(self) -> None:
        # Default success status + error-marker content → error via fallback.
        assert message_is_error(ToolMessage(content="[tool_error] boom", tool_call_id="c"))

    def test_clean_success_not_error(self) -> None:
        assert not message_is_error(ToolMessage(content="正常结果", tool_call_id="c"))
