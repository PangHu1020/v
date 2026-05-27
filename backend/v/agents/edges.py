"""Graph node names and conditional routing helpers."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage

from backend.v.agents.state import CustomerServiceState

ENTER = "enter"
AGENT = "agent"
TOOLS = "tools"
EXIT = "exit"


def route_after_agent(state: CustomerServiceState) -> Literal["tools", "exit"]:
    """Decide whether to dispatch tool calls or finish the turn.

    Inspects the most recent message; if it's an ``AIMessage`` with one or
    more ``tool_calls`` we route to the tools node, otherwise we finish.
    Mirrors LangGraph's prebuilt ``tools_condition`` but ours has explicit
    return strings matching our two destination names.
    """
    messages = state.get("messages", [])
    if not messages:
        return "exit"
    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "exit"
