"""Tool orchestrator: permission check, parallel dispatch, pool limit, timeout, JSON output.

Sits between the guard (which decides *whether* to run) and ToolNode (which
executes). The guard produces ``allowed_calls``; the orchestrator produces
``ToolExecutionResult`` with every call's result wrapped in a uniform JSON
envelope the LLM can read.

JSON envelope (tool result that the agent sees)::

    {"ok": true,  "tool": "search",    "data": "<검색결과…>",  "error": null}
    {"ok": false, "tool": "search",    "data": null,           "error": "…reason…"}

Parallel-safety registry: each tool is registered as parallel-safe (IO-bound,
no session-level mutation visible to other calls) or serial. The LLM does NOT
control scheduling — it's a static property of the tool.

Permission policy (loaded from ``.agent/config.json`` → ``tools.permissions``):
- ``allowed``: execute normally.
- ``ask``:  return a prompt asking the user to confirm before the agent retries.
- ``deny``: block immediately with a clear reason.
Absent entry defaults to ``allowed``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from backend.v.configs.agent_config import ToolPermission
from backend.v.hooks.tool_guard import update_error_counts
from backend.v.utils.logging import get_logger

_log = get_logger("agents.orchestrator")

# ── Constants ──────────────────────────────────────────────────────────────
TOOL_POOL_SIZE: int = 4
"""Max simultaneous tool executions within one batch (semaphore cap)."""

TOOL_BATCH_TIMEOUT_SECONDS: float = 30.0
"""Wall-clock budget for the parallel batch; per-call serial timeout reuses this."""

# Tools that are safe to run concurrently (read-only / idempotent).
# Everything else runs serially. Subagent is always serial regardless of this set.
PARALLEL_SAFE: frozenset[str] = frozenset(
    {
        "search",
        "calculator",
        "recall_memory",
        "load_skill",
    }
)

_SUBAGENT = "subagent"


# ── Result dataclass ───────────────────────────────────────────────────────
@dataclass
class ToolExecutionResult:
    """Outcome of one orchestrated tool-call batch."""

    messages: list[ToolMessage] = field(default_factory=list)
    """All result messages (blocked + executed), JSON-enveloped."""
    new_error_counts: dict[str, int] = field(default_factory=dict)
    """Updated circuit-breaker counts (executed calls only)."""


# ── JSON envelope helpers ──────────────────────────────────────────────────
def _envelope(tool: str, *, ok: bool, data: str | None, error: str | None) -> str:
    """Render the uniform JSON result the LLM reads."""
    return json.dumps(
        {"ok": ok, "tool": tool, "data": data, "error": error},
        ensure_ascii=False,
    )


def _tool_name_for(tool_call_id: str, calls: list[dict[str, Any]]) -> str:
    for tc in calls:
        if tc.get("id") == tool_call_id:
            return tc.get("name", "unknown")
    return "unknown"


def _wrap(msg: ToolMessage, tool: str) -> ToolMessage:
    """Re-wrap a ToolMessage content in the JSON envelope."""
    ok = (msg.status or "success") != "error"
    if ok:
        content = _envelope(tool, ok=True, data=msg.content or "", error=None)
    else:
        content = _envelope(tool, ok=False, data=None, error=msg.content or "")
    return ToolMessage(content=content, tool_call_id=msg.tool_call_id, status=msg.status)


# ── Orchestrator ───────────────────────────────────────────────────────────
class ToolOrchestrator:
    """Execute an allowed tool-call batch with permission check, pool, timeout, and JSON output.

    Args:
        concurrent_node: LangGraph ToolNode for parallel-safe tools.
        serial_node: LangGraph ToolNode for serial tools (subagent).
        permissions: Per-tool permission map from ``.agent/config.json``.
    """

    def __init__(
        self,
        concurrent_node: Any,
        serial_node: Any,
        permissions: dict[str, ToolPermission],
    ) -> None:
        self._concurrent = concurrent_node
        self._serial = serial_node
        self._permissions = permissions

    def _permission(self, tool_name: str) -> ToolPermission:
        return self._permissions.get(tool_name, "allowed")

    async def execute(
        self,
        allowed_calls: list[dict[str, Any]],
        state: dict[str, Any],
        config: Any,
        error_counts: dict[str, int],
    ) -> ToolExecutionResult:
        """Run every call in ``allowed_calls`` through permission → dispatch → timeout.

        Returns blocked messages (permission-denied/ask) + executed messages,
        all wrapped in the JSON envelope. Only executed calls feed
        ``new_error_counts`` (blocked calls must not trip the circuit breaker).
        """
        messages = state.get("messages", [])
        prefix = messages[:-1]  # everything except the pending AIMessage

        blocked: list[ToolMessage] = []
        to_run: list[dict[str, Any]] = []

        for tc in allowed_calls:
            name = tc.get("name", "")
            perm = self._permission(name)
            if perm == "deny":
                _log.warning("orchestrator.tool_denied", tool=name)
                blocked.append(
                    ToolMessage(
                        content=_envelope(
                            name,
                            ok=False,
                            data=None,
                            error=f"工具 {name} 当前不可用，请换其他方式处理。",
                        ),
                        tool_call_id=tc["id"],
                        status="error",
                    )
                )
            elif perm == "ask":
                _log.info("orchestrator.tool_requires_confirm", tool=name)
                blocked.append(
                    ToolMessage(
                        content=_envelope(
                            name,
                            ok=False,
                            data=None,
                            error=f"执行 {name} 前需要客户确认，请先说明并等待同意。",
                        ),
                        tool_call_id=tc["id"],
                        status="error",
                    )
                )
            else:
                to_run.append(tc)

        if not to_run:
            return ToolExecutionResult(messages=blocked, new_error_counts=dict(error_counts))

        # Partition: parallel-safe vs serial (subagent always serial).
        parallel_calls = [
            tc for tc in to_run if tc["name"] in PARALLEL_SAFE and tc["name"] != _SUBAGENT
        ]
        serial_calls = [
            tc for tc in to_run if tc["name"] not in PARALLEL_SAFE or tc["name"] == _SUBAGENT
        ]

        sem = asyncio.Semaphore(TOOL_POOL_SIZE)
        executed: list[ToolMessage] = []

        async def _one(tc: dict[str, Any]) -> list[ToolMessage]:
            async with sem:
                subset = {**state, "messages": [*prefix, AIMessage(content="", tool_calls=[tc])]}
                result = await self._concurrent.ainvoke(subset, config)
                return result.get("messages", [])

        # Parallel batch with overall timeout.
        if parallel_calls and self._concurrent:
            try:
                batches = await asyncio.wait_for(
                    asyncio.gather(*[_one(tc) for tc in parallel_calls]),
                    timeout=TOOL_BATCH_TIMEOUT_SECONDS,
                )
                for batch in batches:
                    executed.extend(batch)
            except TimeoutError:
                _log.warning(
                    "orchestrator.batch_timeout", tools=[tc["name"] for tc in parallel_calls]
                )
                for tc in parallel_calls:
                    executed.append(
                        ToolMessage(
                            content=_envelope(
                                tc["name"],
                                ok=False,
                                data=None,
                                error=f"工具 {tc['name']} 执行超时，请稍后重试。",
                            ),
                            tool_call_id=tc["id"],
                            status="error",
                        )
                    )

        # Serial calls (subagent, or tools outside PARALLEL_SAFE).
        for tc in serial_calls:
            node = self._serial if tc["name"] == _SUBAGENT else self._concurrent
            if not node:
                continue
            try:
                subset = {**state, "messages": [*prefix, AIMessage(content="", tool_calls=[tc])]}
                result = await asyncio.wait_for(
                    node.ainvoke(subset, config),
                    timeout=TOOL_BATCH_TIMEOUT_SECONDS,
                )
                executed.extend(result.get("messages", []))
            except TimeoutError:
                _log.warning("orchestrator.call_timeout", tool=tc["name"])
                executed.append(
                    ToolMessage(
                        content=_envelope(
                            tc["name"],
                            ok=False,
                            data=None,
                            error=f"工具 {tc['name']} 执行超时，请稍后重试。",
                        ),
                        tool_call_id=tc["id"],
                        status="error",
                    )
                )

        # Wrap all executed results in the JSON envelope.
        wrapped = [
            _wrap(msg, _tool_name_for(msg.tool_call_id, to_run))
            if isinstance(msg, ToolMessage)
            else msg
            for msg in executed
        ]

        new_error_counts = update_error_counts(to_run, wrapped, error_counts)
        return ToolExecutionResult(
            messages=blocked + wrapped,
            new_error_counts=new_error_counts,
        )
