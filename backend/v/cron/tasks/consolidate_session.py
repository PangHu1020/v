"""Session consolidation: short-term + medium-term memory writes.

Phase-3 reshape of Phase-2's consolidator. One LLM call still does the
heavy lifting; what changes is the **output shape** (now feeds two
layers) and the **trigger semantics** (mid-session compression also
calls in here).

The 三层温度梯度 design:

- 会话记忆（短期，Redis） — :func:`backend.v.memory.session_memory.write_session_memory`.
  Holds preferences + observations the LLM extracted from this session;
  consumed at session-end promotion (固化) and by the mid-session
  compression node when it re-injects personalization context.
- 事件记忆（中期，PG，30 天）— ``agent.session_memory`` row. Holds
  narrative + intents + key_facts + sentiment + unresolved; consumed at
  next-session ``on_session_start`` injection.

The session-end promotion to long-term ``user_profile`` is owned by
:mod:`backend.v.memory.memory_extractor` and runs only when the session
truly closes (not on every compression).

Returns the new ``agent.session_memory.id`` so callers (e.g., a
``compression_node``) can fetch the row back if they need the rendered
narrative.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg
import redis.asyncio as redis_async
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from backend.v.agents.checkpoints.redis import RedisCheckpointer
from backend.v.cron.tasks.prompts import SESSION_SUMMARIZER_SYSTEM_PROMPT
from backend.v.memory.session_memory import write_session_memory
from backend.v.models.llm_caller import LLMCaller
from backend.v.utils.logging import bind_request, get_logger

_log = get_logger("cron.consolidate_session")

DEFAULT_EVENT_TTL_DAYS = 30


class SessionExtraction(BaseModel):
    """Structured output of the consolidation prompt.

    Two halves: the medium-term ``事件记忆`` columns (narrative …
    unresolved) and the short-term ``会话记忆`` columns (preferences,
    observations). The session-end promotion later decides which of the
    short-term preferences cross over into the long-term ``user_profile``.
    """

    # 事件记忆 (medium-term, PG agent.session_memory)
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

    # 会话记忆 (short-term, Redis session_memory)
    preferences: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Actionable preferences the agent should respect for the rest "
            'of THIS session. Examples: {"preferred_courier": "顺丰", '
            '"language": "普通话"}. Only include items the customer '
            "stated or strongly implied this session."
        ),
    )
    observations: list[str] = Field(
        default_factory=list,
        description=(
            "Free-form short observations about the current state — tone, "
            "urgency, special context. Used by the compression node to "
            "preserve agent persona across compressed turns."
        ),
    )


_SUMMARIZE_SYSTEM_PROMPT = SESSION_SUMMARIZER_SYSTEM_PROMPT


def _format_history(messages: list[BaseMessage]) -> str:
    """Render the messages list as a plain-text transcript for the summarizer."""
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
) -> SessionExtraction | None:
    prompt: list[BaseMessage] = [
        SystemMessage(content=_SUMMARIZE_SYSTEM_PROMPT),
        HumanMessage(content=f"<transcript>\n{transcript}\n</transcript>"),
    ]
    try:
        result = await llm_caller.chat("summary", prompt, structured=SessionExtraction)
    except Exception as exc:
        _log.error("cron.consolidate.llm_failed", error=type(exc).__name__)
        return None
    try:
        payload = (
            json.loads(result.message.content) if isinstance(result.message.content, str) else {}
        )
        return SessionExtraction.model_validate(payload)
    except Exception as exc:
        _log.error("cron.consolidate.parse_failed", error=type(exc).__name__)
        return None


async def consolidate_session(
    ctx: dict[str, Any],
    *,
    session_id: str,
    channel: str | None = None,
    channel_user_id: str | None = None,
) -> str | None:
    """Run the dual-write consolidation.

    1. Read the thread's messages from the Redis checkpointer.
    2. Run one LLM call producing :class:`SessionExtraction`.
    3. Write the medium-term half to ``agent.session_memory`` with a
       30-day ``expires_at``.
    4. Write the short-term half to Redis via
       :func:`write_session_memory`.
    5. Mark the session row ``status = 'consolidated'``.

    Returns the inserted ``agent.session_memory.id`` on success or
    ``None`` if there's nothing to consolidate. Tolerant of LLM / parse
    failures (logs + returns None).
    """
    pool: asyncpg.Pool = ctx["pool"]
    redis: redis_async.Redis = ctx["redis"]
    llm_caller: LLMCaller = ctx["llm_caller"]
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

        token_count = len(transcript)
        async with pool.acquire() as conn:
            row_id = await conn.fetchval(
                "INSERT INTO agent.session_memory "
                "(session_id, summary, token_count, metadata, expires_at) "
                "VALUES ($1, $2, $3, $4, now() + ($5 || ' days')::interval) "
                "RETURNING id",
                session_id,
                extraction.narrative,
                token_count,
                {
                    "intents": extraction.intents,
                    "key_facts": extraction.key_facts,
                    "sentiment": extraction.sentiment,
                    "unresolved": extraction.unresolved,
                },
                str(event_ttl_days),
            )
            await conn.execute(
                "UPDATE agent.session SET status = 'consolidated', "
                "last_activity_at = now() WHERE session_id = $1",
                session_id,
            )

        await write_session_memory(
            redis,
            session_id=session_id,
            preferences=extraction.preferences,
            observations=extraction.observations,
            ttl_seconds=ttl_seconds,
        )

        _log.info(
            "cron.consolidate.persisted",
            row_id=str(row_id),
            intents=len(extraction.intents),
            key_facts=len(extraction.key_facts),
            preferences=len(extraction.preferences),
            observations=len(extraction.observations),
            event_ttl_days=event_ttl_days,
        )
        return str(row_id)
