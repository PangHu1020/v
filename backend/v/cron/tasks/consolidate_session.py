"""Session consolidation: dual-write into working + event memory.

Phase-3 Group H reshape. One LLM call still does the heavy lifting,
but the output shape is now :class:`backend.v.memory.types.ExtractionResult`
(a unified ``profile_updates`` + ``working_memories`` + ``event_memories``)
and the writes split:

- ``working_memories`` → Redis list via
  :func:`backend.v.memory.working.append_working_memory`.
- ``event_memories``   → ``agent.event_memory`` rows via
  :func:`backend.v.memory.event_memory.insert_event_memories` (each
  embedded at write time).
- ``profile_updates``  → IGNORED here; the long-term profile is owned
  by :mod:`backend.v.memory.memory_extractor.promote_to_long_term`,
  which runs only at session end.

The session row's ``status`` is flipped to ``'consolidated'`` so a
re-run is a clean no-op.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg
import redis.asyncio as redis_async
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from backend.v.agents.checkpoints.redis import RedisCheckpointer
from backend.v.cron.tasks.prompts import SESSION_CONSOLIDATION_SYSTEM_PROMPT
from backend.v.memory.event_memory import (
    DEFAULT_EVENT_TTL_DAYS,
    insert_event_memories,
)
from backend.v.memory.types import ExtractionResult
from backend.v.memory.working import append_working_memory
from backend.v.models.llm_caller import LLMCaller
from backend.v.utils.logging import bind_request, get_logger

_log = get_logger("cron.consolidate_session")


def _format_history(messages: list[BaseMessage]) -> str:
    """Render the messages list as a plain-text transcript for the consolidator."""
    parts: list[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, HumanMessage):
            role = "客户"
        elif isinstance(m, AIMessage):
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                names = ", ".join(tc.get("name", "?") for tc in tool_calls)
                parts.append(f"[助手调用工具: {names}]")
                continue
            role = "助手"
        else:
            role = "系统"
        text = (
            m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
        )
        parts.append(f"{role}: {text}")
    return "\n".join(parts)


async def _run_llm_extraction(
    llm_caller: LLMCaller,
    transcript: str,
) -> ExtractionResult | None:
    """Run the LLM and parse the ExtractionResult shape. Returns ``None`` on failure."""
    prompt: list[BaseMessage] = [
        SystemMessage(content=SESSION_CONSOLIDATION_SYSTEM_PROMPT),
        HumanMessage(content=f"<transcript>\n{transcript}\n</transcript>"),
    ]
    try:
        result = await llm_caller.chat("summary", prompt, structured=ExtractionResult)
    except Exception as exc:
        _log.error("cron.consolidate.llm_failed", error=type(exc).__name__)
        return None
    try:
        payload = (
            json.loads(result.message.content) if isinstance(result.message.content, str) else {}
        )
        return ExtractionResult.model_validate(payload)
    except Exception as exc:
        _log.error("cron.consolidate.parse_failed", error=type(exc).__name__)
        return None


async def consolidate_session(
    ctx: dict[str, Any],
    *,
    session_id: str,
    channel: str | None = None,
    channel_user_id: str | None = None,
) -> dict[str, int] | None:
    """Run the dual-write consolidation.

    1. Read the thread's messages from the Redis checkpointer.
    2. Run one LLM call producing :class:`ExtractionResult`.
    3. Append ``working_memories`` to the session's Redis working memory.
    4. Embed + INSERT each ``event_memories`` entry into ``agent.event_memory``
       with a 30-day expires_at.
    5. Mark the session row ``status = 'consolidated'``.

    ``profile_updates`` are deliberately ignored here — they're produced
    by the session-end promotion path (``promote_to_long_term``).

    Returns ``{"working_inserted": int, "events_inserted": int}`` on
    success or ``None`` if there's nothing to consolidate.
    """
    pool: asyncpg.Pool = ctx["pool"]
    redis: redis_async.Redis = ctx["redis"]
    llm_caller: LLMCaller = ctx["llm_caller"]
    embedder: Embeddings = ctx["embedder"]
    ttl_seconds: int = ctx.get("ttl_seconds", 1800)
    event_ttl_days: int = ctx.get("event_ttl_days", DEFAULT_EVENT_TTL_DAYS)

    with bind_request(session_id=session_id, channel=channel, channel_user_id=channel_user_id):
        ckpt = RedisCheckpointer(redis, ttl_seconds=ttl_seconds)
        snap = await ckpt.aget_tuple({"configurable": {"thread_id": session_id}})
        if snap is None:
            _log.info("cron.consolidate.no_checkpoint")
            return None

        messages = snap.checkpoint.get("channel_values", {}).get("messages") or []
        if not messages:
            _log.info("cron.consolidate.no_messages")
            return None

        transcript = _format_history(messages)
        if not transcript.strip():
            _log.info("cron.consolidate.empty_transcript")
            return None

        extraction = await _run_llm_extraction(llm_caller, transcript)
        if extraction is None:
            return None

        # Resolve identity for event_memory rows when the caller didn't
        # supply it. The event_memory table is partitioned by
        # (channel, channel_user_id), so we need it to be correct.
        if channel is None or channel_user_id is None:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT channel, channel_user_id FROM agent.session WHERE session_id = $1",
                    session_id,
                )
            if row is None:
                _log.warning("cron.consolidate.no_identity")
                return None
            channel = channel or row["channel"]
            channel_user_id = channel_user_id or row["channel_user_id"]

        # Working-memory writes are cheap (Redis RPUSH); event-memory
        # writes go through the embedder for each row.
        await append_working_memory(
            redis,
            session_id=session_id,
            entries=extraction.working_memories,
            ttl_seconds=ttl_seconds,
        )
        events_inserted = await insert_event_memories(
            pool,
            channel=channel,
            channel_user_id=channel_user_id,
            session_id=session_id,
            entries=extraction.event_memories,
            embedder=embedder,
            ttl_days=event_ttl_days,
        )

        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE agent.session SET status = 'consolidated', "
                "last_activity_at = now() WHERE session_id = $1",
                session_id,
            )

        _log.info(
            "cron.consolidate.persisted",
            working_inserted=len(extraction.working_memories),
            events_inserted=events_inserted,
            event_ttl_days=event_ttl_days,
        )
        return {
            "working_inserted": len(extraction.working_memories),
            "events_inserted": events_inserted,
        }
