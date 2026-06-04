"""Mid-session compression node (Phase-3 Group H).

Inserted between ``enter`` and ``agent``. When ``state.messages`` token
count exceeds ``compression_threshold_tokens``, drops the oldest messages
keeping the most recent ``compression_keep_recent_messages`` turns verbatim,
and inserts a ``<compressed_history>`` placeholder in their place.

After trimming, the session's working memory is promoted to long-term storage
(user_profile + event_memory) via :func:`backend.v.memory.memory_extractor.promote_to_long_term`
when the required context keys (pool, redis, llm_caller, embedder) are present.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from backend.v.agents.prompts import build_compression_summary
from backend.v.agents.state import CustomerServiceState
from backend.v.memory.prompts import render_session_memory_for_prompt
from backend.v.memory.working import read_working_memory
from backend.v.utils.logging import get_logger
from backend.v.utils.tokens import count_messages_tokens

_log = get_logger("agents.compression")


async def compression_node(
    state: CustomerServiceState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Trim history and consolidate working memory when token threshold fires."""
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

    session_id: str = state.get("session_id") or cfg.get("thread_id") or ""

    # --- 1. Promote working memory to long-term storage ---
    pool = cfg.get("pg_pool")
    redis = cfg.get("redis")
    llm_caller = cfg.get("llm_caller")
    embedder = cfg.get("embedder")

    if session_id and pool and redis and llm_caller and embedder:
        try:
            from backend.v.memory.memory_extractor import promote_to_long_term

            result = await promote_to_long_term(
                {"pool": pool, "redis": redis, "llm_caller": llm_caller, "embedder": embedder},
                session_id=session_id,
            )
            _log.info("agents.compression.memory_promoted", result=result)
        except Exception as exc:
            _log.error("agents.compression.memory_promote_failed", error=type(exc).__name__)

    # --- 2. Reload working memory for the summary block ---
    working_entries = []
    if session_id and redis:
        try:
            working_entries = await read_working_memory(redis, session_id=session_id)
        except Exception as exc:
            _log.warning("agents.compression.working_memory_read_failed", error=type(exc).__name__)
    sm_block = render_session_memory_for_prompt(working_entries)

    # --- 3. Trim messages, inject compressed marker ---
    head = messages[:-keep_recent]
    tail = messages[-keep_recent:]

    removals = [RemoveMessage(id=mid) for m in head if (mid := getattr(m, "id", None))]
    compressed = SystemMessage(
        content=build_compression_summary(narrative=None, session_memory_block=sm_block)
    )

    _log.info(
        "agents.compression.applied",
        messages_before=len(messages),
        kept_recent=keep_recent,
    )
    return {"messages": [*removals, compressed, *tail]}
