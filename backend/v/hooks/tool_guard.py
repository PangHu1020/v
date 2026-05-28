"""Tool-call safety guards (Phase-3 Group D).

Two mechanisms:

**Dead-loop detection** — tracks a fingerprint (``tool_name:canonical_args``)
for every tool call in the current session. When the same fingerprint
appears 3 times a warning is logged; at 5 times the guard sets
``state["force_handoff"] = True`` so the next ``agent_node`` invocation
injects a ``transfer_to_human`` tool call instead of asking the LLM.

**Per-(tool, session) circuit breaker** — counts how many times a tool
has returned an error in this session. When the count reaches
``CIRCUIT_OPEN_THRESHOLD`` (default 3) the tool is blocked for the rest
of the session and the guard sets ``force_handoff = True``.

Both mechanisms are state-based (no Redis) so they work without
additional infrastructure and survive within a single session's
checkpoint chain.
"""

from __future__ import annotations

import json
from typing import Any

from backend.v.utils.logging import get_logger

_log = get_logger("hooks.tool_guard")

LOOP_WARN_THRESHOLD = 3
LOOP_STOP_THRESHOLD = 5
CIRCUIT_OPEN_THRESHOLD = 3


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


def evaluate_tool_calls(
    tool_calls: list[dict[str, Any]],
    fingerprints: list[str],
    error_counts: dict[str, int],
) -> tuple[list[str], bool]:
    """Evaluate all pending tool calls and return ``(new_fps, force_handoff)``.

    ``new_fps`` is the list of fingerprints for the current batch (to be
    appended to state). ``force_handoff`` is ``True`` if any call should
    be blocked.
    """
    new_fps: list[str] = []
    force = False
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args") or {}
        fp = compute_fingerprint(name, args)
        new_fps.append(fp)
        count, stop = check_loop(fingerprints, fp)
        if stop:
            _log.error("tool_guard.loop_stop", tool=name, count=count + 1)
            force = True
        elif count >= LOOP_WARN_THRESHOLD:
            _log.warning("tool_guard.loop_warn", tool=name, count=count + 1)
        if is_circuit_open(error_counts, name):
            _log.error("tool_guard.circuit_open", tool=name)
            force = True
    return new_fps, force


def update_error_counts(
    tool_calls: list[dict[str, Any]],
    tool_messages: list[Any],
    error_counts: dict[str, int],
) -> dict[str, int]:
    """Increment error counts for tools whose ToolMessage looks like an error."""
    updated = dict(error_counts)
    id_to_name = {tc["id"]: tc.get("name", "") for tc in tool_calls}
    for msg in tool_messages:
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id and is_tool_error(getattr(msg, "content", "")):
            name = id_to_name.get(tool_call_id, "")
            if name:
                updated[name] = updated.get(name, 0) + 1
    return updated
