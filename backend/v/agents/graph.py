"""Compile the Phase-1 LangGraph."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from backend.v.agents.edges import AGENT, ENTER, EXIT
from backend.v.agents.nodes import agent_node, enter_node, exit_node
from backend.v.agents.state import CustomerServiceState


def build_graph(checkpointer: BaseCheckpointSaver):
    """Compile the linear customer-service graph.

    Args:
        checkpointer: Working-memory backed checkpointer (Redis in Phase-1).

    Returns:
        A compiled graph ready to ``ainvoke`` per turn. Callers must pass
        ``config={"configurable": {"thread_id": session_id, "llm_caller": ...}}``.
    """
    graph = StateGraph(CustomerServiceState)
    graph.add_node(ENTER, enter_node)
    graph.add_node(AGENT, agent_node)
    graph.add_node(EXIT, exit_node)
    graph.set_entry_point(ENTER)
    graph.add_edge(ENTER, AGENT)
    graph.add_edge(AGENT, EXIT)
    graph.add_edge(EXIT, END)
    return graph.compile(checkpointer=checkpointer)
