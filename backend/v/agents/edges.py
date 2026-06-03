"""Graph node names and conditional routing helpers."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage

from backend.v.agents.state import CustomerServiceState

ENTER = "enter"
COMPRESS = "compress"
INTENT = "intent"
AGENT = "agent"
AGENT_FAST = "agent_fast"  # general intent: no tools bound
TOOLS = "tools"
REFLECT = "reflect"
EXIT = "exit"

# Intents that need hallucination reflection
_REFLECTION_INTENTS = frozenset({"refund", "logistics"})


def route_after_intent(
    state: CustomerServiceState,
) -> Literal["agent", "agent_fast"]:
    """Route general intent to tool-free fast path; all others to full agent."""
    if state.get("intent") == "general":
        return "agent_fast"
    return "agent"


def route_after_agent(state: CustomerServiceState) -> Literal["tools", "reflect", "exit"]:
    """Route to tools if there are pending tool_calls, else to reflect or exit."""
    messages = state.get("messages", [])
    if not messages:
        return "exit"
    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    # Only run reflection for high-risk intents
    if state.get("needs_reflection"):
        return "reflect"
    return "exit"


def route_after_reflect(state: CustomerServiceState) -> Literal["agent", "exit"]:
    """Retry the agent turn when reflection failed; otherwise exit."""
    if state.get("reflection_failed"):
        return "agent"
    return "exit"
