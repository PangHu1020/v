"""Graph node names and conditional routing helpers."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage

from backend.v.agents.state import CustomerServiceState

ENTER = "enter"
COMPRESS = "compress"
INTENT = "intent"
AGENT = "agent"
TOOLS = "tools"
REFLECT = "reflect"
EXIT = "exit"


def route_after_agent(state: CustomerServiceState) -> Literal["tools", "exit"]:
    """Route to tools if the last AIMessage has tool_calls, else exit."""
    messages = state.get("messages", [])
    if not messages:
        return "exit"
    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "exit"


def route_after_reflect(state: CustomerServiceState) -> Literal["agent", "exit"]:
    """Retry the agent turn when reflection failed; otherwise exit."""
    if state.get("reflection_failed"):
        return "agent"
    return "exit"
