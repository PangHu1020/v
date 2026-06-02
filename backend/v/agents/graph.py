"""Compile the customer-service LangGraph.

Phase-3 Group E adds intent classification and reflection:

    enter -> compress -> intent -> agent -> [tool_calls?] -> tools -> agent
                                                                         |
                                                                      reflect
                                                                     /      \\
                                                                  agent    exit -> END
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
    COMPRESS,
    ENTER,
    EXIT,
    INTENT,
    REFLECT,
    TOOLS,
    route_after_agent,
    route_after_reflect,
)
from backend.v.agents.intent_reflect import intent_node, reflection_node
from backend.v.agents.nodes import AGENT_TOOLS, agent_node, enter_node, exit_node
from backend.v.agents.state import CustomerServiceState
from backend.v.hooks.tool_guard import evaluate_tool_calls, update_error_counts


def _make_guarded_tools_node(bound_tools: list[Any]):
    from langgraph.prebuilt import ToolNode

    _tool_node = ToolNode(bound_tools)

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
            error_msgs = [
                ToolMessage(
                    content="[tool_guard] 工具调用被安全机制阻止",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
            return {
                "messages": error_msgs,
                "tool_fingerprints": fingerprints + new_fps,
            }

        result = await _tool_node.ainvoke(state, config)
        new_error_counts = update_error_counts(tool_calls, result.get("messages", []), error_counts)
        return {
            **result,
            "tool_fingerprints": fingerprints + new_fps,
            "tool_error_counts": new_error_counts,
        }

    return _guarded


def build_graph(checkpointer: BaseCheckpointSaver):
    """Compile the agent graph with intent routing + reflection."""
    bound_tools = list(AGENT_TOOLS)

    async def _agent(state: CustomerServiceState, config: RunnableConfig) -> dict[str, Any]:
        return await agent_node(state, config, tools=bound_tools)

    graph = StateGraph(CustomerServiceState)
    graph.add_node(ENTER, enter_node)
    graph.add_node(COMPRESS, compression_node)
    graph.add_node(INTENT, intent_node)
    graph.add_node(AGENT, _agent)
    graph.add_node(TOOLS, _make_guarded_tools_node(bound_tools))
    graph.add_node(REFLECT, reflection_node)
    graph.add_node(EXIT, exit_node)
    graph.set_entry_point(ENTER)
    graph.add_edge(ENTER, COMPRESS)
    graph.add_edge(COMPRESS, INTENT)
    graph.add_edge(INTENT, AGENT)
    graph.add_conditional_edges(AGENT, route_after_agent, {"tools": TOOLS, "exit": REFLECT})
    graph.add_edge(TOOLS, AGENT)
    graph.add_conditional_edges(REFLECT, route_after_reflect, {"agent": AGENT, "exit": EXIT})
    graph.add_edge(EXIT, END)
    return graph.compile(checkpointer=checkpointer)
