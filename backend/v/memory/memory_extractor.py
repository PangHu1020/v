"""Long-term memory promotion (Phase-3 三层温度梯度第三层入口).

固化（Consolidation step）: at session end, read the short-term
``会话记忆`` (Redis) and ask DeepSeek flash with structured Pydantic
output to:

1. Update ``agent.user_profile`` JSONB with whatever **long-term-useful**
   subset of this session's preferences should cross over (the LLM sees
   the existing profile + the session preferences and decides which keys
   are tendency-bound vs session-bound).
2. Extract a list of standalone episodic facts about the customer
   (Chinese sentences like "客户偏好顺丰快递", "曾投诉物流延误"),
   embed them via Qwen ``text-embedding-v4`` (1024-dim Matryoshka), and
   INSERT into ``agent.memory_episodes`` so
   :func:`backend.v.tools.recall_memory` can semantically retrieve them.

Phase-3 trigger change: this no longer fires after every
``consolidate_session``. Only the session-end pathway calls it, because:

- Mid-session compression should not pollute long-term memory with
  preferences that may yet be contradicted in the same session.
- Promoting on every compression doubles LLM cost for marginal benefit.

Idempotent on partial failures: profile upsert is its own transaction,
episode inserts are independent rows. A retry sees the existing profile
and merges the same way.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from backend.v.memory.prompts import MEMORY_EXTRACTOR_SYSTEM_PROMPT
from backend.v.memory.session_memory import (
    delete_session_memory,
    read_session_memory,
)
from backend.v.models.llm_caller import LLMCaller
from backend.v.utils.logging import bind_request, get_logger

_log = get_logger("memory.extractor")


class Episode(BaseModel):
    """One free-form memory fact about the customer."""

    content: str = Field(
        min_length=1,
        description="A standalone Chinese sentence the agent can recall later.",
    )
    importance: int = Field(
        default=3,
        ge=1,
        le=5,
        description="1 = trivia, 5 = critical (compliance, safety).",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Short labels for retrieval / filtering.",
    )


class ExtractionOutput(BaseModel):
    """LLM-produced merge of (existing profile, session preferences) into
    both a refreshed long-term profile and a list of episodes."""

    profile: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The updated user_profile JSON. Preserve the existing fields "
            "unless the new session contradicts them; add new ones only "
            "when justified. Empty if nothing changed."
        ),
    )
    episodes: list[Episode] = Field(
        default_factory=list,
        description="Episodic memories worth long-term recall.",
    )


_PROMOTION_SYSTEM_PROMPT = MEMORY_EXTRACTOR_SYSTEM_PROMPT


def _build_prompt(
    existing_profile: dict[str, Any],
    session_memory: dict[str, Any],
) -> list[BaseMessage]:
    prefs_json = json.dumps(session_memory.get("preferences", {}), ensure_ascii=False)
    obs_json = json.dumps(session_memory.get("observations", []), ensure_ascii=False)
    return [
        SystemMessage(content=_PROMOTION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "<existing_profile>\n"
                f"{json.dumps(existing_profile, ensure_ascii=False, indent=2)}\n"
                "</existing_profile>\n\n"
                "<session_memory>\n"
                f"  <preferences>{prefs_json}</preferences>\n"
                f"  <observations>{obs_json}</observations>\n"
                "</session_memory>"
            )
        ),
    ]


async def _read_session_identity(
    pool: asyncpg.Pool,
    session_id: str,
) -> tuple[str, str] | None:
    """Look up ``(channel, channel_user_id)`` for a session_id."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT channel, channel_user_id FROM agent.session WHERE session_id = $1",
            session_id,
        )
    if row is None:
        return None
    return row["channel"], row["channel_user_id"]


async def _read_existing_profile(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT profile FROM agent.user_profile WHERE channel = $1 AND channel_user_id = $2",
            channel,
            channel_user_id,
        )
    if row is None or row["profile"] is None:
        return {}
    return dict(row["profile"])


async def _upsert_user_profile(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
    profile: dict[str, Any],
) -> None:
    """Upsert ``profile`` for the given identity. Replaces any prior row."""
    if not profile:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO agent.user_profile (channel, channel_user_id, profile) "
            "VALUES ($1, $2, $3) "
            "ON CONFLICT (channel, channel_user_id) DO UPDATE "
            "SET profile = EXCLUDED.profile, updated_at = now()",
            channel,
            channel_user_id,
            profile,
        )


async def _embed_and_insert_episodes(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
    episodes: list[Episode],
    embedder: Embeddings,
) -> int:
    """Embed each episode and INSERT one row per item. Returns inserted count."""
    if not episodes:
        return 0
    texts = [e.content for e in episodes]
    vectors = await embedder.aembed_documents(texts)
    inserted = 0
    async with pool.acquire() as conn:
        for episode, vec in zip(episodes, vectors, strict=True):
            await conn.execute(
                "INSERT INTO agent.memory_episodes "
                "(channel, channel_user_id, content, embedding, metadata) "
                "VALUES ($1, $2, $3, $4, $5)",
                channel,
                channel_user_id,
                episode.content,
                vec,
                {"importance": episode.importance, "tags": episode.tags},
            )
            inserted += 1
    return inserted


async def extract_session_memory(
    ctx: dict[str, Any],
    *,
    session_id: str,
) -> dict[str, int] | None:
    """Promote this session's short-term 会话记忆 to long-term storage.

    Reads the Redis session_memory record (preferences + observations),
    asks the LLM to filter for long-term-useful items, and writes the
    surviving items into ``agent.user_profile`` (structured) and
    ``agent.memory_episodes`` (vectorized). On success, drops the Redis
    record so a subsequent re-run is a clean no-op.

    Returns ``{"profile_updated": 0|1, "episodes_inserted": N}`` on
    success, ``None`` if there's nothing to promote (no session row, no
    short-term record, or LLM/parse failure).
    """
    pool: asyncpg.Pool = ctx["pool"]
    redis = ctx["redis"]
    llm_caller: LLMCaller = ctx["llm_caller"]
    embedder: Embeddings = ctx["embedder"]

    with bind_request(session_id=session_id):
        identity = await _read_session_identity(pool, session_id)
        if identity is None:
            _log.info("memory.extractor.no_session_row")
            return None
        channel, channel_user_id = identity

        sm = await read_session_memory(redis, session_id=session_id)
        if sm is None:
            _log.info("memory.extractor.no_session_memory")
            return None

        existing = await _read_existing_profile(
            pool, channel=channel, channel_user_id=channel_user_id
        )

        prompt = _build_prompt(existing, sm)
        try:
            result = await llm_caller.chat("memory_extract", prompt, structured=ExtractionOutput)
        except Exception as exc:
            _log.error("memory.extractor.llm_failed", error=type(exc).__name__)
            return None

        try:
            payload = (
                json.loads(result.message.content)
                if isinstance(result.message.content, str)
                else {}
            )
            extracted = ExtractionOutput.model_validate(payload)
        except Exception as exc:
            _log.error("memory.extractor.parse_failed", error=type(exc).__name__)
            return None

        profile_updated = 0
        if extracted.profile and extracted.profile != existing:
            await _upsert_user_profile(
                pool,
                channel=channel,
                channel_user_id=channel_user_id,
                profile=extracted.profile,
            )
            profile_updated = 1

        episodes_inserted = await _embed_and_insert_episodes(
            pool,
            channel=channel,
            channel_user_id=channel_user_id,
            episodes=extracted.episodes,
            embedder=embedder,
        )

        # On a successful promotion, drop the Redis short-term record so a
        # later cron run re-firing on the same session is a clean no-op.
        await delete_session_memory(redis, session_id=session_id)

        _log.info(
            "memory.extractor.done",
            profile_updated=profile_updated,
            episodes_inserted=episodes_inserted,
        )
        return {
            "profile_updated": profile_updated,
            "episodes_inserted": episodes_inserted,
        }
