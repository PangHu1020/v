"""Mid-session compression node (Phase-3 Group H).

When token threshold fires:
1. LLM extracts working_memories + event_memories from the about-to-be-dropped
   history. Both are written to their stores. profile_updates is deferred to
   session-end (session still active — no profile changes mid-conversation).
2. Message history is trimmed; compressed marker injected with the freshly
   written working-memory block so the next turn still sees the summary.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from backend.v.agents.prompts import build_compression_summary
from backend.v.agents.state import CustomerServiceState
from backend.v.memory.prompts import (
    MID_SESSION_EXTRACTION_SYSTEM_PROMPT,
    render_session_memory_for_prompt,
)
from backend.v.memory.types import ExtractionResult
from backend.v.utils.logging import get_logger
from backend.v.utils.tokens import count_messages_tokens

_log = get_logger("agents.compression")


async def compression_node(
    state: CustomerServiceState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Extract memories from dropped history, then trim the message list."""
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
    pool = cfg.get("pg_pool")
    redis = cfg.get("redis")
    llm_caller = cfg.get("llm_caller")
    embedder = cfg.get("embedder")

    head = messages[:-keep_recent]
    tail = messages[-keep_recent:]

    # --- 1. LLM extraction over the about-to-be-dropped head ---
    extracted = ExtractionResult()
    if llm_caller and head:
        from langchain_core.messages import AIMessage

        lines: list[str] = []
        for m in head:
            if isinstance(m, HumanMessage):
                lines.append(f"客户：{m.content}")
            elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                lines.append(f"助理：{m.content}")

        if lines:
            prompt = [
                SystemMessage(content=MID_SESSION_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=f"<conversation>\n{chr(10).join(lines)}\n</conversation>"),
            ]
            try:
                res = await llm_caller.chat("memory_extract", prompt, structured=ExtractionResult)
                if isinstance(res.parsed, ExtractionResult):
                    extracted = res.parsed
            except Exception as exc:
                _log.error("agents.compression.extract_failed", error=type(exc).__name__)

    # --- 2. Write working_memories to Redis ---
    if session_id and redis and extracted.working_memories:
        try:
            from backend.v.memory.working import append_working_memory

            ttl = cfg.get("cache_ttl_seconds", 1800)
            await append_working_memory(
                redis,
                session_id=session_id,
                entries=extracted.working_memories,
                ttl_seconds=ttl,
            )
        except Exception as exc:
            _log.error("agents.compression.working_write_failed", error=type(exc).__name__)

    # --- 3. Write event_memories to Postgres ---
    if pool and embedder and session_id and extracted.event_memories:
        try:
            from backend.v.memory.event_memory import insert_event_memories

            identity = state.get("channel"), state.get("channel_user_id")
            if all(identity):
                await insert_event_memories(
                    pool,
                    channel=identity[0],
                    channel_user_id=identity[1],
                    session_id=session_id,
                    entries=extracted.event_memories,
                    embedder=embedder,
                )
        except Exception as exc:
            _log.error("agents.compression.event_write_failed", error=type(exc).__name__)

    # --- 4. Trim messages, inject compressed marker ---
    from backend.v.memory.working import read_working_memory

    working_entries = []
    if session_id and redis:
        try:
            working_entries = await read_working_memory(redis, session_id=session_id)
        except Exception as exc:
            _log.warning("agents.compression.working_read_failed", error=type(exc).__name__)

    sm_block = render_session_memory_for_prompt(working_entries)
    removals = [RemoveMessage(id=mid) for m in head if (mid := getattr(m, "id", None))]
    compressed = SystemMessage(
        content=build_compression_summary(
            conversation_state=extracted.conversation_state,
            session_memory_block=sm_block,
        )
    )

    _log.info(
        "agents.compression.applied",
        messages_before=len(messages),
        kept_recent=keep_recent,
        working_written=len(extracted.working_memories),
        events_written=len(extracted.event_memories),
        has_state=extracted.conversation_state is not None,
    )
    return {"messages": [*removals, compressed, *tail]}
