"""Graph state for the customer-service agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from backend.v.memory.types import MemoryEntry


class CustomerServiceState(TypedDict, total=False):
    """LangGraph state passed between nodes.

    Attributes:
        messages: Conversation history; ``add_messages`` reducer appends new
            ``BaseMessage`` instances.
        session_id: Mint-on-30-min-silence identifier for the conversation.
        channel: Channel slug (e.g. ``wecom_aibot``).
        channel_user_id: External per-channel user id.
        user_profile: Long-term ``user_profile`` row injected at session start.
        recent_events: Medium-term ``agent.event_memory`` dicts loaded at
            session start (newest first).
        working_memory: :class:`MemoryEntry` list from this session's Redis
            working memory.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    channel: str
    channel_user_id: str
    user_profile: dict[str, Any] | None
    recent_events: list[dict[str, Any]]
    working_memory: list[MemoryEntry]
    # Phase-3 Group D: tool safety guards
    tool_fingerprints: list[str]
    tool_error_counts: dict[str, int]
    # Phase-3 Group E: intent routing + reflection
    intent: str
    needs_reflection: bool  # set by intent_node; True only for refund/logistics
    reflection_retries: int
    reflection_failed: bool
