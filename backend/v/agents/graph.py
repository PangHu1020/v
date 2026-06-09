"""Compile the customer-service LangGraph.

Topology (post-refactor):

    enter -> compress -> intent ──┬── agent_fast (general) -> exit
                                  └── agent (refund/logistics/complaint)
                                        │
                              [tool_calls?] -> tools (parallel) -> agent
                                        │
                              [needs_reflection?] -> reflect ──┬── agent (retry)
                                                               └── exit
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
from backend.v.agents.intent_reflect import intent_node, reflection_node
from backend.v.agents.nodes import AGENT_TOOLS, agent_fast_node, agent_node, enter_node, exit_node
from backend.v.agents.state import CustomerServiceState
from backend.v.hooks.tool_guard import evaluate_tool_calls, update_error_counts

_SUBAGENT_NAME = "subagent"


def _make_guarded_tools_node(bound_tools: list[Any]):
    """ToolNode wrapper that:
    - runs non-subagent tools concurrently via asyncio.gather
    - runs subagent serially after the rest (avoids nested LLM concurrency risk)
    - enforces dead-loop + circuit-breaker guards
    """
    from langgraph.prebuilt import ToolNode

    # Separate tool sets for concurrent vs serial execution
    _concurrent_tools = [t for t in bound_tools if getattr(t, "name", None) != _SUBAGENT_NAME]
    _serial_tools = [t for t in bound_tools if getattr(t, "name", None) == _SUBAGENT_NAME]
    _concurrent_node = ToolNode(_concurrent_tools) if _concurrent_tools else None
    _serial_node = ToolNode(_serial_tools) if _serial_tools else None

    async def _guarded(state: CustomerServiceState, config: RunnableConfig) -> dict[str, Any]:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        if not (isinstance(last, AIMessage) and getattr(last, "tool_calls", None)):
            return {}

        fingerprints = list(state.get("tool_fingerprints") or [])
        error_counts = dict(state.get("tool_error_counts") or {})
        tool_calls = last.tool_calls
        new_fps, force = evaluate_tool_calls(tool_calls, fingerprints, error_counts)

        if force:
            return {
                "messages": [
                    ToolMessage(
                        content="[tool_guard] 工具调用被安全机制阻止",
                        tool_call_id=tc["id"],
                    )
                    for tc in tool_calls
                ],
                "tool_fingerprints": fingerprints + new_fps,
            }

        # Split pending calls by concurrency class
        concurrent_calls = [tc for tc in tool_calls if tc["name"] != _SUBAGENT_NAME]
        serial_calls = [tc for tc in tool_calls if tc["name"] == _SUBAGENT_NAME]

        all_messages: list = []

        # Run concurrent tools in parallel
        if concurrent_calls and _concurrent_node:
            subset_state = {
                **state,
                "messages": [*messages[:-1], AIMessage(content="", tool_calls=concurrent_calls)],
            }
            result = await _concurrent_node.ainvoke(subset_state, config)
            all_messages.extend(result.get("messages", []))

        for call in serial_calls:
            if _serial_node:
                subset_state = {
                    **state,
                    "messages": [*messages[:-1], AIMessage(content="", tool_calls=[call])],
                }
                result = await _serial_node.ainvoke(subset_state, config)
                all_messages.extend(result.get("messages", []))

        new_error_counts = update_error_counts(tool_calls, all_messages, error_counts)
        return {
            "messages": all_messages,
            "tool_fingerprints": fingerprints + new_fps,
            "tool_error_counts": new_error_counts,
        }

    return _guarded


def build_graph(checkpointer: BaseCheckpointSaver, *, extra_tools: list[Any] | None = None):
    """Compile the agent graph with intent routing + conditional reflection.

    Args:
        checkpointer: LangGraph checkpointer (Redis hot path in production).
        extra_tools: Dynamically-discovered tools to bind on top of the
            built-in ``AGENT_TOOLS`` — e.g. MCP server tools resolved at
            startup. Bound to both the LLM and the guarded ToolNode.
    """
    bound_tools = list(AGENT_TOOLS) + list(extra_tools or [])

    async def _agent(state: CustomerServiceState, config: RunnableConfig) -> dict[str, Any]:
        return await agent_node(state, config, tools=bound_tools)

    graph = StateGraph(CustomerServiceState)
    graph.add_node(ENTER, enter_node)
    graph.add_node(COMPRESS, compression_node)
    graph.add_node(INTENT, intent_node)
    graph.add_node(AGENT, _agent)
    graph.add_node(AGENT_FAST, agent_fast_node)
    graph.add_node(TOOLS, _make_guarded_tools_node(bound_tools))
    graph.add_node(REFLECT, reflection_node)
    graph.add_node(EXIT, exit_node)

    graph.set_entry_point(ENTER)
    graph.add_edge(ENTER, COMPRESS)
    graph.add_edge(COMPRESS, INTENT)
    graph.add_conditional_edges(
        INTENT, route_after_intent, {"agent": AGENT, "agent_fast": AGENT_FAST}
    )
    graph.add_conditional_edges(
        AGENT, route_after_agent, {"tools": TOOLS, "reflect": REFLECT, "exit": EXIT}
    )
    graph.add_edge(AGENT_FAST, EXIT)
    graph.add_edge(TOOLS, AGENT)
    graph.add_conditional_edges(REFLECT, route_after_reflect, {"agent": AGENT, "exit": EXIT})
    graph.add_edge(EXIT, END)
    return graph.compile(checkpointer=checkpointer)
