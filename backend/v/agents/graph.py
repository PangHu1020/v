"""Compile the customer-service LangGraph."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from backend.v.agents.edges import AGENT, ENTER, EXIT, TOOLS, route_after_agent
from backend.v.agents.nodes import AGENT_TOOLS, agent_node, enter_node, exit_node
from backend.v.agents.state import CustomerServiceState


def build_graph(
    checkpointer: BaseCheckpointSaver,
    *,
    extra_tools: list[Any] | None = None,
):
    """Compile the agent graph with the tool-call loop.

    Uses closures (not functools.partial) for node wrappers so LangGraph's
    signature inspector sees plain ``(state, config)`` functions and
    correctly threads the config through every node.
    """
    bound_tools = list(AGENT_TOOLS) + list(extra_tools or [])

    async def _agent(state: CustomerServiceState, config: RunnableConfig) -> dict[str, Any]:
        return await agent_node(state, config, tools=bound_tools)

    graph = StateGraph(CustomerServiceState)
    graph.add_node(ENTER, enter_node)
    graph.add_node(AGENT, _agent)
    graph.add_node(TOOLS, ToolNode(bound_tools))
    graph.add_node(EXIT, exit_node)
    graph.set_entry_point(ENTER)
    graph.add_edge(ENTER, AGENT)
    graph.add_conditional_edges(AGENT, route_after_agent, {"tools": TOOLS, "exit": EXIT})
    graph.add_edge(TOOLS, AGENT)
    graph.add_edge(EXIT, END)
    return graph.compile(checkpointer=checkpointer)
