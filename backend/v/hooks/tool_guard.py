"""Tool-call safety guards (Phase-3 Group D).

Three mechanisms:

**Per-turn call budget** — caps how many tool calls one customer turn may
issue. A turn begins at the latest ``HumanMessage``; once the cumulative tool
calls since then would exceed :data:`TURN_MAX_TOOL_CALLS`, the guard blocks
further calls so the agent must answer from what it already retrieved. This is
the defence against *near-duplicate* thrash — a model that re-searches with the
query reworded each time slips past fingerprint dedup (every fingerprint
differs), but the raw count still stops it. Bounds token blow-up deterministically.

**Dead-loop detection** — tracks a fingerprint (``tool_name:canonical_args``)
for every tool call in the current session. When the same fingerprint
appears 3 times a warning is logged; at 5 times the guard forces a block so
the next ``agent_node`` invocation answers instead of re-calling. Catches
*exact* repeats (identical args).

**Per-(tool, session) circuit breaker** — counts how many times a tool
has returned an error in this session. When the count reaches
``CIRCUIT_OPEN_THRESHOLD`` (default 3) the tool is blocked for the rest
of the session.

All mechanisms are state-based (no Redis) so they work without additional
infrastructure and survive within a single session's checkpoint chain.
"""

from __future__ import annotations

import json
from typing import Any

from backend.v.utils.logging import get_logger

_log = get_logger("hooks.tool_guard")

LOOP_WARN_THRESHOLD = 3
LOOP_STOP_THRESHOLD = 5
CIRCUIT_OPEN_THRESHOLD = 3
TURN_MAX_TOOL_CALLS = 8
"""Max tool calls one customer turn may issue before the guard forces the agent
to answer. Sized from eval: well-behaved cases peak around 5 calls/turn, while
runaway re-search loops hit 16+/turn — 8 leaves headroom for legitimate
multi-constraint fan-out (one search per sub-need) yet halts the thrash."""


def count_tool_calls_since_last_human(messages: list[Any]) -> int:
    """Count tool calls issued in the current turn (since the last HumanMessage).

    Walks ``messages`` from the end, summing each AIMessage's ``tool_calls``
    until a HumanMessage is reached (the turn boundary). Uses duck-typing on
    ``msg.type`` so this module stays free of langchain imports.
    """
    count = 0
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            break
        tcs = getattr(msg, "tool_calls", None)
        if tcs:
            count += len(tcs)
    return count


def exceeds_turn_budget(prior_messages: list[Any], pending_count: int) -> bool:
    """Whether running ``pending_count`` more tool calls breaches the turn cap.

    ``prior_messages`` is the message history *excluding* the pending AIMessage
    (its calls are passed via ``pending_count``). Returns ``True`` when the
    already-issued calls this turn plus the pending batch would exceed
    :data:`TURN_MAX_TOOL_CALLS`.
    """
    prior = count_tool_calls_since_last_human(prior_messages)
    return prior + pending_count > TURN_MAX_TOOL_CALLS


def compute_fingerprint(tool_name: str, args: dict[str, Any]) -> str:
    """Canonical string key for a (tool, args) pair."""
    return f"{tool_name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"


def check_loop(
    fingerprints: list[str],
    new_fp: str,
) -> tuple[int, bool]:
    """Return ``(count_so_far, should_stop)``.

    ``count_so_far`` is how many times ``new_fp`` already appears in
    ``fingerprints`` (before the current call is appended). ``should_stop``
    is ``True`` when the count reaches :data:`LOOP_STOP_THRESHOLD`.
    """
    count = fingerprints.count(new_fp)
    return count, count >= LOOP_STOP_THRESHOLD


def is_circuit_open(error_counts: dict[str, int], tool_name: str) -> bool:
    """Return ``True`` when ``tool_name`` has hit the error threshold."""
    return error_counts.get(tool_name, 0) >= CIRCUIT_OPEN_THRESHOLD


def is_tool_error(content: Any) -> bool:
    """Heuristic: a ToolMessage content that starts with ``[`` is an error."""
    if isinstance(content, str):
        return content.startswith("[") or "error" in content.lower()
    return False


def message_is_error(msg: Any) -> bool:
    """Whether a ToolMessage represents a failed tool call.

    ``status="error"`` (set by LangGraph's ToolNode when a tool *raises*, and by
    the guard on blocked calls) is authoritative — locale- and wording-independent.

    Otherwise falls back to the content heuristic. This fallback is REQUIRED, not
    optional: tools that signal failure by *returning* a marker string (MCP's
    ``[tool_error]``, ``[subagent_error]``) come back with the default
    ``status="success"`` because they didn't raise — only the content tells them
    apart from a legit empty-result success like ``（暂无相关历史记忆）``. So we
    never treat ``status="success"`` as authoritatively-not-error.
    """
    if getattr(msg, "status", None) == "error":
        return True
    return is_tool_error(getattr(msg, "content", ""))


# ── Block reasons ─────────────────────────────────────────────────────────
# Each blocked tool call gets a *specific* reason + next-step guidance, so the
# LLM can adapt (change args / switch tool / answer from what it has) instead of
# blindly retrying against an opaque "blocked" wall. The text is the ToolMessage
# content the model reads; ``[tool_guard]`` prefix marks it as a guard signal
# (reflection skips these as non-facts).
BLOCK_LOOP = "loop"
BLOCK_CIRCUIT = "circuit"
BLOCK_BUDGET = "budget"

_BLOCK_TEMPLATES: dict[str, str] = {
    BLOCK_LOOP: (
        "[tool_guard] 已用完全相同的参数调用 {tool} {count} 次，不再重复执行。"
        "换一组不同的参数再试，或直接用已经检索到的结果回答客户。"
    ),
    BLOCK_CIRCUIT: (
        "[tool_guard] 工具 {tool} 本次会话已连续失败 {count} 次，已暂停使用。"
        "改用其他工具，或在确实拿不到数据时转人工，不要再调用它。"
    ),
    BLOCK_BUDGET: (
        "[tool_guard] 本轮工具调用已达上限（{cap} 次），不再执行新的调用。"
        "请基于本轮已获取的结果直接作答，不要再发起检索。"
    ),
}


def block_message(reason: str, *, tool: str = "", count: int = 0, cap: int = 0) -> str:
    """Render the LLM-facing guidance text for a block ``reason``."""
    template = _BLOCK_TEMPLATES.get(reason)
    if template is None:  # unknown reason → safe generic fallback
        return "[tool_guard] 工具调用被安全机制阻止。"
    return template.format(tool=tool, count=count, cap=cap)


def evaluate_tool_calls(
    tool_calls: list[dict[str, Any]],
    fingerprints: list[str],
    error_counts: dict[str, int],
) -> tuple[list[str], list[dict[str, Any] | None]]:
    """Evaluate pending tool calls per-call for loop / circuit blocks.

    Returns ``(new_fps, reasons)`` where ``new_fps`` is the fingerprint of each
    call (appended to state) and ``reasons`` is aligned 1:1 with ``tool_calls``:
    ``None`` means the call is allowed, otherwise a dict
    ``{"reason": <BLOCK_*>, "tool": str, "count": int}`` describing why THAT
    call is blocked. The turn-budget block is applied separately by the caller
    (it needs message history), and takes precedence over a ``None`` here.
    """
    new_fps: list[str] = []
    reasons: list[dict[str, Any] | None] = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args") or {}
        fp = compute_fingerprint(name, args)
        new_fps.append(fp)
        reason: dict[str, Any] | None = None
        count, stop = check_loop(fingerprints, fp)
        if is_circuit_open(error_counts, name):
            # Circuit takes precedence: the tool itself is unhealthy.
            _log.error("tool_guard.circuit_open", tool=name)
            reason = {"reason": BLOCK_CIRCUIT, "tool": name, "count": error_counts.get(name, 0)}
        elif stop:
            _log.error("tool_guard.loop_stop", tool=name, count=count + 1)
            reason = {"reason": BLOCK_LOOP, "tool": name, "count": count + 1}
        elif count >= LOOP_WARN_THRESHOLD:
            _log.warning("tool_guard.loop_warn", tool=name, count=count + 1)
        reasons.append(reason)
    return new_fps, reasons


def update_error_counts(
    tool_calls: list[dict[str, Any]],
    tool_messages: list[Any],
    error_counts: dict[str, int],
) -> dict[str, int]:
    """Increment error counts for tools whose ToolMessage represents a failure.

    Uses :func:`message_is_error` (status-first, content-fallback) so genuine
    failures count toward the circuit breaker while empty-result successes do not.
    """
    updated = dict(error_counts)
    id_to_name = {tc["id"]: tc.get("name", "") for tc in tool_calls}
    for msg in tool_messages:
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id and message_is_error(msg):
            name = id_to_name.get(tool_call_id, "")
            if name:
                updated[name] = updated.get(name, 0) + 1
    return updated
