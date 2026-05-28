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

from langchain_core.messages import ToolMessage

from backend.v.hooks.tool_guard import (
    CIRCUIT_OPEN_THRESHOLD,
    LOOP_STOP_THRESHOLD,
    LOOP_WARN_THRESHOLD,
    check_loop,
    compute_fingerprint,
    evaluate_tool_calls,
    is_circuit_open,
    is_tool_error,
    update_error_counts,
)


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
        assert is_circuit_open({"recall_memory": CIRCUIT_OPEN_THRESHOLD - 1}, "recall_memory") is False

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
    def test_first_call_no_force(self) -> None:
        calls = [{"id": "c1", "name": "recall_memory", "args": {"q": "a"}}]
        new_fps, force = evaluate_tool_calls(calls, [], {})
        assert len(new_fps) == 1
        assert force is False

    def test_loop_stop_forces_handoff(self) -> None:
        fp = compute_fingerprint("recall_memory", {"q": "a"})
        existing = [fp] * LOOP_STOP_THRESHOLD
        calls = [{"id": "c1", "name": "recall_memory", "args": {"q": "a"}}]
        new_fps, force = evaluate_tool_calls(calls, existing, {})
        assert force is True
        assert new_fps == [fp]

    def test_warn_does_not_force(self) -> None:
        fp = compute_fingerprint("recall_memory", {"q": "a"})
        existing = [fp] * LOOP_WARN_THRESHOLD
        calls = [{"id": "c1", "name": "recall_memory", "args": {"q": "a"}}]
        _, force = evaluate_tool_calls(calls, existing, {})
        assert force is False

    def test_circuit_open_forces_handoff(self) -> None:
        calls = [{"id": "c1", "name": "recall_memory", "args": {"q": "a"}}]
        _, force = evaluate_tool_calls(
            calls,
            [],
            {"recall_memory": CIRCUIT_OPEN_THRESHOLD},
        )
        assert force is True

    def test_one_bad_call_taints_batch(self) -> None:
        good = {"id": "c1", "name": "recall_memory", "args": {"q": "ok"}}
        bad = {"id": "c2", "name": "recall_memory", "args": {"q": "loop"}}
        loop_fp = compute_fingerprint("recall_memory", {"q": "loop"})
        existing = [loop_fp] * LOOP_STOP_THRESHOLD
        new_fps, force = evaluate_tool_calls([good, bad], existing, {})
        assert force is True
        assert len(new_fps) == 2


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
