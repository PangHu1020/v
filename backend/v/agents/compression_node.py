"""Mid-session compression node (memory V2).

When the token threshold fires:
1. One LLM call extracts a structured ``ConversationState`` (for seamless
   continuation) plus working/event sentences from the about-to-be-dropped
   history.
2. Those sentences are folded into the session's **Redis working memory** only.
   No Postgres write happens here — under memory V2 every durable write is
   centralised at session-end (``promote_to_long_term``), which reads working
   memory as its input. Keeping mid-session purely Redis-local means the hot
   path never blocks on PG and there is exactly one durable-write site.
3. The message history is trimmed; a ``<compressed_history>`` marker carrying
   the conversation state + current working memory is injected so the next turn
   continues seamlessly.
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
    """Extract memories into Redis working memory, then trim the message list."""
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
    redis = cfg.get("redis")
    llm_caller = cfg.get("llm_caller")

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

    # --- 2. Fold extracted sentences into Redis working memory (no PG write) ---
    # Both working- and event-kind sentences go to Redis; session-end
    # consolidation reads working memory and routes them to the durable tiers.
    folded = [*extracted.working_memories, *extracted.event_memories]
    if session_id and redis and folded:
        try:
            from backend.v.memory.working import append_working_memory

            ttl = cfg.get("cache_ttl_seconds", 1800)
            await append_working_memory(
                redis,
                session_id=session_id,
                entries=folded,
                ttl_seconds=ttl,
            )
        except Exception as exc:
            _log.error("agents.compression.working_write_failed", error=type(exc).__name__)

    # --- 3. Trim messages, inject compressed marker ---
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
        folded_to_working=len(folded),
        has_state=extracted.conversation_state is not None,
    )
    return {"messages": [*removals, compressed, *tail]}
