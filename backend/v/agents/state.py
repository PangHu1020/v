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


class CustomerServiceState(TypedDict, total=False):
    """LangGraph state passed between nodes.

    Attributes:
        messages: Conversation history; ``add_messages`` reducer appends new
            ``BaseMessage`` instances. Includes a system prompt prepended by
            ``on_session_start`` plus the user's turn(s) and the AI replies.
        session_id: Mint-on-30-min-silence identifier for the conversation.
        channel: Channel slug (``wecom`` / ``feishu``).
        channel_user_id: External per-channel user id.
        user_profile: Long-term ``user_profile`` row injected at session start.
            ``None`` when the user has no stored profile yet.
        interrupt_payload: Reserved for Phase-2 ``transfer_to_human`` handoff
            metadata. Phase-1 nodes leave it untouched.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    channel: str
    channel_user_id: str
    user_profile: dict[str, Any] | None
    interrupt_payload: dict[str, Any] | None
