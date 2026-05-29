"""Graph state for the customer-service agent.

The state is a :class:`TypedDict` whose ``messages`` key uses the LangGraph
``add_messages`` reducer for monotonic append. All other keys are
overwritten on each node update following the standard LangGraph pattern;
nodes return only the keys they want to update.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from backend.v.memory.types import MemoryEntry


class CustomerServiceState(TypedDict, total=False):
    """LangGraph state passed between nodes.

    Attributes:
        messages: Conversation history; ``add_messages`` reducer appends new
            ``BaseMessage`` instances. Includes a system prompt prepended by
            ``enter_node`` plus the user's turn(s) and the AI replies.
        session_id: Mint-on-30-min-silence identifier for the conversation.
        channel: Channel slug (``wecom`` / ``feishu`` / ``wecom_aibot``).
        channel_user_id: External per-channel user id.
        user_profile: Long-term ``user_profile`` row injected at session start.
            ``None`` when the user has no stored profile yet.
        recent_events: List of medium-term ``agent.event_memory`` dicts loaded
            at session start (newest first; rendered into ``<recent_events>``
            by ``enter_node``).
        working_memory: List of :class:`MemoryEntry` from this session's Redis
            working memory (rendered into ``<session_memory>``). Empty list
            on the first turn before any consolidation has fired.
        interrupt_payload: Reserved for ``transfer_to_human`` handoff metadata.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    channel: str
    channel_user_id: str
    user_profile: dict[str, Any] | None
    recent_events: list[dict[str, Any]]
    working_memory: list[MemoryEntry]
    interrupt_payload: dict[str, Any] | None
    # Phase-3 Group D: tool safety guards
    tool_fingerprints: list[str]
    """Fingerprints of every tool call in this session (``tool:args``).
    Used by the dead-loop detector in :mod:`backend.v.hooks.tool_guard`."""
    tool_error_counts: dict[str, int]
    """Per-tool error counts for the circuit breaker."""
    force_handoff: bool
    """Set by the tool guard when a loop or circuit-open condition is
    detected. ``agent_node`` checks this and injects a
    ``transfer_to_human`` call instead of asking the LLM."""
    # Phase-3 Group E: intent routing + reflection
    intent: str
    """Classified intent from ``intent_node``: refund / logistics / complaint / general."""
    reflection_retries: int
    """How many reflection retries have fired this turn."""
    reflection_failed: bool
    """Set by ``reflection_node`` when the check fails; cleared on retry."""
