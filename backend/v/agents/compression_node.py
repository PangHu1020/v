"""Mid-session compression node (Phase-3 Group H).

Inserted between ``enter`` and ``agent``. When ``state.messages`` token
count exceeds ``compression_threshold_tokens``, drops the oldest messages
keeping the most recent ``compression_keep_recent_messages`` turns verbatim,
and inserts a ``<compressed_history>`` placeholder in their place.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from backend.v.agents.prompts import build_compression_summary
from backend.v.agents.state import CustomerServiceState
from backend.v.utils.logging import get_logger
from backend.v.utils.tokens import count_messages_tokens

_log = get_logger("agents.compression")


async def compression_node(
    state: CustomerServiceState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Trim history when token count exceeds the configured threshold."""
    cfg = config.get("configurable", {}) if config else {}
    threshold: int = int(cfg.get("compression_threshold_tokens", 0))
    if threshold <= 0:
        return {}

    messages: list[BaseMessage] = list(state.get("messages", []))
    keep_recent: int = max(1, int(cfg.get("compression_keep_recent_messages", 4)))
    if not messages or len(messages) <= keep_recent:
        return {}

    model_name = cfg.get("token_model")
    if count_messages_tokens(messages, model_name) < threshold:
        return {}

    head = messages[:-keep_recent]
    tail = messages[-keep_recent:]

    removals = [RemoveMessage(id=mid) for m in head if (mid := getattr(m, "id", None))]
    compressed = SystemMessage(
        content=build_compression_summary(narrative=None, session_memory_block="")
    )

    _log.info(
        "agents.compression.applied",
        messages_before=len(messages),
        kept_recent=keep_recent,
    )
    return {"messages": [*removals, compressed, *tail]}
