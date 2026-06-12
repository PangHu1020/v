"""Graph node names and conditional routing helpers."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage

from backend.v.agents.state import CustomerServiceState

ENTER = "enter"
COMPRESS = "compress"
INTENT = "intent"
CLARIFY = "clarify"
AGENT = "agent"
AGENT_FAST = "agent_fast"  # chitchat: no tools bound
TOOLS = "tools"
REFLECT = "reflect"
EXIT = "exit"

# Intents that need hallucination reflection
_REFLECTION_INTENTS = frozenset({"refund", "logistics"})


def route_after_intent(
    state: CustomerServiceState,
) -> Literal["clarify", "agent", "agent_fast"]:
    """Route by the derived intent state.

    - ambiguous (under clarify cap) → clarify, which then hands to agent.
    - pure chitchat single intent → tool-free fast path.
    - everything else (mix, refund, logistics, complaint, general, or an
      ambiguous turn that exhausted the clarify cap) → full agent.
    """
    if state.get("needs_clarify"):
        return "clarify"
    if state.get("intent") == "chitchat" and not state.get("is_mix"):
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
