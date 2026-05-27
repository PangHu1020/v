"""Long-term memory writer (Phase-2 P3).

Reads the most recent ``agent.session_memory`` row for a session and uses
DeepSeek flash with structured Pydantic output to:

1. Produce an updated ``user_profile`` JSONB by intelligently merging the
   session's findings with the existing stored profile (the LLM gets both
   as input so it can decide what to keep, replace, or add).
2. Extract a list of standalone episodic memories (Chinese sentences
   like "客户对发货速度敏感，曾投诉延误"), embed them via Qwen
   ``text-embedding-v4`` (1024-dim Matryoshka), and INSERT into
   ``agent.memory_episodes`` so :func:`backend.v.tools.recall_memory`
   can semantically retrieve them on future turns.

Triggered as an ARQ delayed job right after
:func:`backend.v.cron.tasks.consolidate_session.consolidate_session`
finishes; runs in the background so neither the agent's reactive turn
nor the consolidation itself wait on this slower work.

Idempotent on partial failures: profile upsert is its own transaction,
episode inserts are independent rows. A retry sees the existing profile
and merges the same way; episodes may end up with mild duplicates which
the recency-weighted recall surfaces only the newest of.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

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
    """LLM-produced merge of (existing profile, session findings) into both
    a refreshed profile and a list of episodes."""

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


_EXTRACT_SYSTEM_PROMPT = (
    "你是一名长期记忆抽取助手。"
    "输入：客户的现有画像（user_profile）+ 一次会话的总结（含 narrative / intents / "
    "key_facts / sentiment / unresolved）。"
    "输出：① 更新后的 user_profile（保留旧字段，仅在新信息明确支持时增改）；"
    "② episodes 列表（每条是一句独立的中文事实，例如「客户偏好夜间收货」「投诉过物流延误」）。"
    "若没有可加入长期记忆的信息，episodes 留空。"
    "不要编造信息，宁可少而准。"
)


def _build_prompt(
    existing_profile: dict[str, Any],
    session_narrative: str,
    session_meta: dict[str, Any],
) -> list[BaseMessage]:
    return [
        SystemMessage(content=_EXTRACT_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "现有画像（JSON）：\n"
                f"```json\n{json.dumps(existing_profile, ensure_ascii=False, indent=2)}\n```\n\n"
                "本次会话总结：\n"
                f"narrative: {session_narrative}\n"
                f"intents: {session_meta.get('intents', [])}\n"
                f"key_facts: {session_meta.get('key_facts', [])}\n"
                f"sentiment: {session_meta.get('sentiment', 'neutral')}\n"
                f"unresolved: {session_meta.get('unresolved', [])}\n"
            )
        ),
    ]


async def _read_session_memory(
    pool: asyncpg.Pool,
    session_id: str,
) -> tuple[str, dict[str, Any]] | None:
    """Return ``(summary, metadata)`` of the latest session_memory row, or ``None``."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT summary, metadata FROM agent.session_memory "
            "WHERE session_id = $1 ORDER BY created_at DESC LIMIT 1",
            session_id,
        )
    if row is None:
        return None
    metadata = row["metadata"] or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return row["summary"], dict(metadata)


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
    """Pull the latest session_memory row and promote it into long-term storage.

    Returns ``{"profile_updated": 0|1, "episodes_inserted": N}`` on
    success, or ``None`` if there's nothing to extract (no session_memory,
    no session row, or the LLM failed).
    """
    pool: asyncpg.Pool = ctx["pool"]
    llm_caller: LLMCaller = ctx["llm_caller"]
    embedder: Embeddings = ctx["embedder"]

    with bind_request(session_id=session_id):
        identity = await _read_session_identity(pool, session_id)
        if identity is None:
            _log.info("memory.extractor.no_session_row")
            return None
        channel, channel_user_id = identity

        sm = await _read_session_memory(pool, session_id)
        if sm is None:
            _log.info("memory.extractor.no_session_memory")
            return None
        summary, meta = sm

        existing = await _read_existing_profile(
            pool, channel=channel, channel_user_id=channel_user_id
        )

        prompt = _build_prompt(existing, summary, meta)
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

        # Merge: LLM returns the FULL desired profile (it sees the existing
        # one). If empty, leave the stored profile alone.
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

        _log.info(
            "memory.extractor.done",
            profile_updated=profile_updated,
            episodes_inserted=episodes_inserted,
        )
        return {
            "profile_updated": profile_updated,
            "episodes_inserted": episodes_inserted,
        }
