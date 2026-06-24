"""Compile the customer-service LangGraph.

Topology (intent V2 — probability-distribution routing):

    enter -> compress -> intent -> {clarify->agent | agent_fast->exit | agent}
                              [tool_calls?] -> tools (parallel) -> agent
                              [needs_reflection?] -> reflect -> {agent retry | exit}

intent_node emits a 5-way probability distribution and DERIVES the route:
top-1 below its per-category threshold OR top-2 within margin -> ambiguous ->
clarify (capped by max_clarify_turns, then falls back to top-1); top-2 both
clear their thresholds -> mix -> agent + reflect; pure chitchat -> agent_fast.
clarify_node only sets a transient directive (answer top-1 + confirm intent),
consumed by agent_node for one call, never persisted to messages.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from backend.v.agents.compression_node import compression_node
from backend.v.agents.edges import (
    AGENT,
    AGENT_FAST,
    CLARIFY,
    COMPRESS,
    ENTER,
    EXIT,
    INTENT,
    REFLECT,
    TOOLS,
    route_after_agent,
    route_after_intent,
    route_after_reflect,
)
from backend.v.agents.intent_reflect import clarify_node, intent_node, reflection_node
from backend.v.agents.nodes import AGENT_TOOLS, agent_fast_node, agent_node, enter_node, exit_node
from backend.v.agents.orchestrator import ToolOrchestrator
from backend.v.agents.state import CustomerServiceState
from backend.v.hooks.tool_guard import (
    BLOCK_BUDGET,
    TURN_MAX_TOOL_CALLS,
    block_message,
    evaluate_tool_calls,
    exceeds_turn_budget,
)
from backend.v.utils.logging import get_logger

_log = get_logger("agents.graph")

_SUBAGENT_NAME = "subagent"

GRAPH_RECURSION_LIMIT = 40
"""LangGraph super-step ceiling per ``ainvoke`` (one customer turn). A healthy
turn under the per-turn tool budget (TURN_MAX_TOOL_CALLS=8) needs ~20 steps, so
40 is a backstop that only trips on pathological non-tool loops (e.g.
agent↔reflect ping-pong). Tripping raises GraphRecursionError → caught by the
bus consumer's handler guard → DLQ. The per-turn budget is the primary defence;
this is depth-in-depth."""


def _make_guarded_tools_node(bound_tools: list[Any], tool_permissions: dict | None = None):
    """ToolNode wrapper: guard checks (loop/circuit/budget) + orchestrated execution.

    The guard decides *whether* to run each call; the :class:`ToolOrchestrator`
    decides *how* (permission check, parallel pool, timeout, JSON envelope).
    """
    from langgraph.prebuilt import ToolNode

    _concurrent_tools = [t for t in bound_tools if getattr(t, "name", None) != _SUBAGENT_NAME]
    _serial_tools = [t for t in bound_tools if getattr(t, "name", None) == _SUBAGENT_NAME]
    _concurrent_node = ToolNode(_concurrent_tools) if _concurrent_tools else None
    _serial_node = ToolNode(_serial_tools) if _serial_tools else None
    _orchestrator = ToolOrchestrator(_concurrent_node, _serial_node, tool_permissions or {})

    async def _guarded(state: CustomerServiceState, config: RunnableConfig) -> dict[str, Any]:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        if not (isinstance(last, AIMessage) and getattr(last, "tool_calls", None)):
            return {}

        fingerprints = list(state.get("tool_fingerprints") or [])
        error_counts = dict(state.get("tool_error_counts") or {})
        tool_calls = last.tool_calls
        new_fps, reasons = evaluate_tool_calls(tool_calls, fingerprints, error_counts)

        # Per-turn budget applies to the whole batch: once this turn's calls
        # would exceed the cap, every pending call is blocked with the BUDGET
        # reason (overrides per-call loop/circuit reasons). Defends against
        # reworded-query thrash that slips past fingerprint dedup.
        budget_block = exceeds_turn_budget(messages[:-1], len(tool_calls))
        if budget_block:
            _log.warning(
                "tool_guard.turn_budget_exceeded",
                pending=len(tool_calls),
                cap=TURN_MAX_TOOL_CALLS,
            )

        # Build a blocked ToolMessage (status=error) for each call the guard
        # stops, with a SPECIFIC reason so the LLM can adapt; collect the rest
        # to actually execute.
        blocked_messages: list = []
        allowed_calls: list = []
        for tc, reason in zip(tool_calls, reasons, strict=True):
            if budget_block:
                content = block_message(BLOCK_BUDGET, cap=TURN_MAX_TOOL_CALLS)
            elif reason is not None:
                content = block_message(
                    reason["reason"], tool=reason["tool"], count=reason["count"]
                )
            else:
                allowed_calls.append(tc)
                continue
            blocked_messages.append(
                ToolMessage(content=content, tool_call_id=tc["id"], status="error")
            )

        # Nothing left to run (all blocked by guard) → return block messages only.
        if not allowed_calls:
            return {
                "messages": blocked_messages,
                "tool_fingerprints": fingerprints + new_fps,
            }

        # Orchestrator: permission check → parallel/serial dispatch → timeout → JSON envelope.
        orch_result = await _orchestrator.execute(allowed_calls, state, config, error_counts)
        return {
            "messages": blocked_messages + orch_result.messages,
            "tool_fingerprints": fingerprints + new_fps,
            "tool_error_counts": orch_result.new_error_counts,
        }

    return _guarded


def build_graph(
    checkpointer: BaseCheckpointSaver,
    *,
    extra_tools: list[Any] | None = None,
    tool_permissions: dict | None = None,
):
    """Compile the agent graph with intent routing + conditional reflection.

    Args:
        checkpointer: LangGraph checkpointer (Redis hot path in production).
        extra_tools: Dynamically-discovered tools to bind on top of the
            built-in ``AGENT_TOOLS`` — e.g. MCP server tools resolved at
            startup. Bound to both the LLM and the guarded ToolNode.
        tool_permissions: Per-tool execution policy from ``.agent/config.json``
            (``allowed`` / ``ask`` / ``deny``). Empty / absent → all allowed.
    """
    bound_tools = list(AGENT_TOOLS) + list(extra_tools or [])

    async def _agent(state: CustomerServiceState, config: RunnableConfig) -> dict[str, Any]:
        return await agent_node(state, config, tools=bound_tools)

    graph = StateGraph(CustomerServiceState)
    graph.add_node(ENTER, enter_node)
    graph.add_node(COMPRESS, compression_node)
    graph.add_node(INTENT, intent_node)
    graph.add_node(CLARIFY, clarify_node)
    graph.add_node(AGENT, _agent)
    graph.add_node(AGENT_FAST, agent_fast_node)
    graph.add_node(TOOLS, _make_guarded_tools_node(bound_tools, tool_permissions))
    graph.add_node(REFLECT, reflection_node)
    graph.add_node(EXIT, exit_node)

    graph.set_entry_point(ENTER)
    graph.add_edge(ENTER, COMPRESS)
    graph.add_edge(COMPRESS, INTENT)
    graph.add_conditional_edges(
        INTENT,
        route_after_intent,
        {"clarify": CLARIFY, "agent": AGENT, "agent_fast": AGENT_FAST},
    )
    # Clarify only sets a transient directive, then hands to the full agent.
    graph.add_edge(CLARIFY, AGENT)
    graph.add_conditional_edges(
        AGENT, route_after_agent, {"tools": TOOLS, "reflect": REFLECT, "exit": EXIT}
    )
    graph.add_edge(AGENT_FAST, EXIT)
    graph.add_edge(TOOLS, AGENT)
    graph.add_conditional_edges(REFLECT, route_after_reflect, {"agent": AGENT, "exit": EXIT})
    graph.add_edge(EXIT, END)
    return graph.compile(checkpointer=checkpointer)
