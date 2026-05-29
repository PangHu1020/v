"""Mid-session compression node (Phase-3 Group H).

Inserted between ``enter`` and ``agent`` in the graph. On every turn it
counts tokens in ``state.messages``; if the count exceeds
``compression_threshold_tokens`` it:

1. Calls :func:`backend.v.cron.tasks.consolidate_session.consolidate_session`
   inline (synchronous from the graph's perspective). The consolidator
   writes new working-memory entries to Redis and persistent
   event-memory entries to PG (with embeddings).
2. Replaces the older messages with a single ``SystemMessage`` carrying
   a ``<compressed_history>`` marker plus the freshly-loaded working
   memory, while keeping the most recent
   ``compression_keep_recent_messages`` turns verbatim.

The replacement uses :class:`langchain_core.messages.RemoveMessage` IDs
so the LangGraph ``add_messages`` reducer correctly deletes-then-appends
rather than appending the summary on top of the existing history.

When tokens are below the threshold the node is a no-op.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from backend.v.agents.prompts import build_compression_summary
from backend.v.agents.state import CustomerServiceState
from backend.v.memory.prompts import render_session_memory_for_prompt
from backend.v.memory.working import read_working_memory
from backend.v.utils.logging import bind_request, get_logger
from backend.v.utils.tokens import count_messages_tokens

_log = get_logger("agents.compression")


def _summary_block(session_memory_block: str) -> str:
    """Thin alias kept for the call site below.

    Real implementation in :mod:`backend.v.agents.prompts`.
    """
    return build_compression_summary(narrative=None, session_memory_block=session_memory_block)


async def compression_node(
    state: CustomerServiceState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Compress when ``messages`` token count exceeds the configured threshold.

    No-op when below threshold, when no consolidate callable is wired in,
    or when there are too few messages to bother (we always keep at
    least ``keep_recent`` trailing turns intact).
    """
    cfg = config.get("configurable", {}) if config else {}
    threshold: int = int(cfg.get("compression_threshold_tokens", 0))
    if threshold <= 0:
        return {}

    consolidate = cfg.get("consolidate_callable")
    if consolidate is None:
        return {}

    messages: list[BaseMessage] = list(state.get("messages", []))
    if not messages:
        return {}

    keep_recent: int = max(1, int(cfg.get("compression_keep_recent_messages", 4)))
    if len(messages) <= keep_recent:
        return {}

    model_name = cfg.get("token_model")
    total_tokens = count_messages_tokens(messages, model_name)
    if total_tokens < threshold:
        return {}

    session_id: str = state.get("session_id") or cfg.get("thread_id") or ""
    channel = state.get("channel")
    channel_user_id = state.get("channel_user_id")
    if not session_id:
        _log.warning("agents.compression.no_session_id")
        return {}

    with bind_request(channel=channel, channel_user_id=channel_user_id, session_id=session_id):
        ctx = cfg.get("consolidate_ctx") or {}
        try:
            consolidation_result = await consolidate(
                ctx,
                session_id=session_id,
                channel=channel,
                channel_user_id=channel_user_id,
            )
        except Exception as exc:
            _log.error("agents.compression.consolidate_failed", error=type(exc).__name__)
            return {}

        if not consolidation_result:
            _log.info("agents.compression.no_consolidation")
            return {}

        # Re-load working memory after consolidation so the summary
        # reflects what was just written. The consolidator appends to
        # the same list, so a fresh read picks up everything including
        # entries from earlier in the session.
        redis = ctx.get("redis")
        working_entries = []
        if redis is not None:
            working_entries = await read_working_memory(redis, session_id=session_id)
        sm_block = render_session_memory_for_prompt(working_entries)

        compressed = SystemMessage(content=_summary_block(sm_block))

        # Drop everything except the trailing ``keep_recent`` turns.
        head = messages[:-keep_recent]
        tail = messages[-keep_recent:]

        removals: list[BaseMessage] = []
        for m in head:
            mid = getattr(m, "id", None)
            if mid:
                removals.append(RemoveMessage(id=mid))

        update: dict[str, Any] = {"messages": [*removals, compressed, *tail]}
        _log.info(
            "agents.compression.applied",
            tokens_before=total_tokens,
            messages_before=len(messages),
            messages_after=len(update["messages"]) - len(removals),
            kept_recent=keep_recent,
            consolidation_result=consolidation_result,
        )
        return update
