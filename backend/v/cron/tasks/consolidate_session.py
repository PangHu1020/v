"""Session-memory consolidation.

Phase-1 left the working-memory hot path (Redis checkpointer with 30-min
TTL) writing nothing to ``agent.session_memory`` — when the TTL fired or
the customer went silent, the session evaporated. This task closes that
gap.

Triggered in two ways:

- :func:`backend.app.bus.worker` schedules a delayed ARQ job at
  ``TTL - 60s`` whenever a session reaches a context-length threshold
  or naturally idles toward expiry (Phase-2 P2 hook addition).
- A periodic ARQ cron sweeps Redis for sessions whose ``last_seen`` is
  older than 25 minutes and consolidates them (catch-all for missed
  schedules across worker restarts).

The summary is produced by ``LLMCaller`` with ``role="summary"``
(DeepSeek flash) using a structured Pydantic output. It includes a
short narrative + key facts (intents, decisions, customer state) so the
``memory_extractor`` (Phase-2 P3) can promote them into long-term
``user_profile`` / ``memory_episodes`` later.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg
import redis.asyncio as redis_async
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from backend.v.agents.checkpoints.redis import RedisCheckpointer
from backend.v.models.llm_caller import LLMCaller
from backend.v.utils.logging import bind_request, get_logger

_log = get_logger("cron.consolidate_session")


class SessionSummary(BaseModel):
    """Structured output of the consolidate prompt.

    Designed so the Phase-2 P3 ``memory_extractor`` can map fields
    directly into ``agent.user_profile`` / ``agent.memory_episodes``
    without re-prompting the LLM.
    """

    narrative: str = Field(description="One paragraph summary of the session, in Chinese.")
    intents: list[str] = Field(
        default_factory=list,
        description="Top-level intents the customer expressed (e.g., 退款, 物流查询).",
    )
    key_facts: list[str] = Field(
        default_factory=list,
        description="Concrete facts worth remembering (订单号, 偏好, 投诉点).",
    )
    sentiment: str = Field(
        default="neutral",
        description="Customer's overall sentiment: positive / neutral / negative.",
    )
    unresolved: list[str] = Field(
        default_factory=list,
        description="Any items left unresolved at the end of the session.",
    )


_SUMMARIZE_SYSTEM_PROMPT = (
    "你是一名客服会话归档助手。给定一段客服与客户的对话，提取关键信息以便长期记忆。"
    "用简洁、客观的中文产出。narrative 不超过 120 字。"
    "key_facts 是结构化的事实，不是评价。"
    "若信息不足，相关字段留空，不要编造。"
)


def _format_history(messages: list[BaseMessage]) -> str:
    """Render the messages list as a plain-text transcript for the summarizer."""
    parts: list[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue  # Skip the system prompt itself; not part of the conversation.
        if isinstance(m, HumanMessage):
            role = "客户"
        elif isinstance(m, AIMessage):
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                # Tool calls don't read well in transcripts; summarize them.
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


async def consolidate_session(
    ctx: dict[str, Any],
    *,
    session_id: str,
    channel: str | None = None,
    channel_user_id: str | None = None,
) -> str | None:
    """Read the session's messages, summarize via LLM, persist to PG.

    Returns the inserted row's ``id`` (str) on success, or ``None`` if the
    session has no checkpoint (already consolidated or never existed).
    Idempotent on missing data; tolerant of summarization failures (logs
    and returns None rather than raising, so a stuck task doesn't kill
    the worker).
    """
    pool: asyncpg.Pool = ctx["pool"]
    redis: redis_async.Redis = ctx["redis"]
    llm_caller: LLMCaller = ctx["llm_caller"]

    with bind_request(session_id=session_id, channel=channel, channel_user_id=channel_user_id):
        ckpt = RedisCheckpointer(redis, ttl_seconds=ctx.get("ttl_seconds", 1800))
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

        prompt: list[BaseMessage] = [
            SystemMessage(content=_SUMMARIZE_SYSTEM_PROMPT),
            HumanMessage(content=f"对话记录：\n\n{transcript}"),
        ]
        try:
            result = await llm_caller.chat("summary", prompt, structured=SessionSummary)
        except Exception as exc:
            _log.error("cron.consolidate.llm_failed", error=type(exc).__name__)
            return None

        # Structured output round-trip: LLMCaller wraps the parsed model
        # as an AIMessage whose .content is its JSON; rehydrate.
        try:
            payload = (
                json.loads(result.message.content)
                if isinstance(result.message.content, str)
                else {}
            )
            summary = SessionSummary.model_validate(payload)
        except Exception as exc:
            _log.error("cron.consolidate.parse_failed", error=type(exc).__name__)
            return None

        token_count = len(transcript)
        async with pool.acquire() as conn:
            row_id = await conn.fetchval(
                "INSERT INTO agent.session_memory "
                "(session_id, summary, token_count, metadata) "
                "VALUES ($1, $2, $3, $4) RETURNING id",
                session_id,
                summary.narrative,
                token_count,
                {
                    "intents": summary.intents,
                    "key_facts": summary.key_facts,
                    "sentiment": summary.sentiment,
                    "unresolved": summary.unresolved,
                },
            )
            await conn.execute(
                "UPDATE agent.session SET status = 'consolidated', "
                "last_activity_at = now() WHERE session_id = $1",
                session_id,
            )

        _log.info(
            "cron.consolidate.persisted",
            row_id=str(row_id),
            intents=len(summary.intents),
            key_facts=len(summary.key_facts),
        )
        return str(row_id)
