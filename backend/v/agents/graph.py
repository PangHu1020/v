"""Compile the customer-service LangGraph.

Phase-2 P0 introduces a tool-call loop: ``agent`` may emit ``transfer_to_human``
tool calls which dispatch through ``ToolNode``; the tool either returns a
plain result (and the agent continues) or invokes ``interrupt(...)`` and
suspends the graph for human handoff.

    enter -> agent -> [tool_calls?] -> tools -> agent -> ... -> exit -> END
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from backend.v.agents.edges import AGENT, ENTER, EXIT, TOOLS, route_after_agent
from backend.v.agents.nodes import AGENT_TOOLS, agent_node, enter_node, exit_node
from backend.v.agents.state import CustomerServiceState


def build_graph(checkpointer: BaseCheckpointSaver):
    """Compile the agent graph with the tool-call loop.

    Args:
        checkpointer: Working-memory checkpointer (Redis hot path; durable
            Postgres path is used by suspended threads after handoff).

    Returns:
        A compiled graph ready to ``ainvoke`` per turn. Callers pass
        ``config={"configurable": {"thread_id": session_id, "llm_caller": ...}}``.
    """
    graph = StateGraph(CustomerServiceState)
    graph.add_node(ENTER, enter_node)
    graph.add_node(AGENT, agent_node)
    graph.add_node(TOOLS, ToolNode(AGENT_TOOLS))
    graph.add_node(EXIT, exit_node)
    graph.set_entry_point(ENTER)
    graph.add_edge(ENTER, AGENT)
    graph.add_conditional_edges(AGENT, route_after_agent, {"tools": TOOLS, "exit": EXIT})
    graph.add_edge(TOOLS, AGENT)
    graph.add_edge(EXIT, END)
    return graph.compile(checkpointer=checkpointer)
