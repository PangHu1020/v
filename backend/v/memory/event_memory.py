"""Event memory: medium-term persistent sentence store (PG ``agent.event_memory``).

Replaces the Phase-2 multi-field ``agent.session_memory`` AND the
Phase-2 ``agent.memory_episodes``. One row = one
:class:`backend.v.memory.types.MemoryEntry` + an embedding column.

Two read paths consume this table:

- ``on_session_start`` injects the most-recent N rows for cross-session
  continuity (no embedding needed).
- The ``recall_memory`` tool runs vector ANN against the same rows
  (lives in :mod:`backend.v.tools.recall_memory`).

The single write path is :func:`insert_event_memories`, called by the
consolidation flow with vectors already computed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import asyncpg
from langchain_core.embeddings import Embeddings

from backend.v.memory.types import MemoryCandidate, MemoryEntry

DEFAULT_EVENT_TTL_DAYS = 30


async def insert_event_memories(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
    session_id: str | None,
    entries: list[MemoryEntry],
    embedder: Embeddings,
    ttl_days: int = DEFAULT_EVENT_TTL_DAYS,
) -> int:
    """Embed each entry's content and INSERT one row per item.

    Returns the count actually inserted. Empty entries → no embedding
    round-trip, returns 0.
    """
    if not entries:
        return 0
    contents = [e.content for e in entries]
    vectors = await embedder.aembed_documents(contents)

    inserted = 0
    async with pool.acquire() as conn:
        async with conn.transaction():
            for entry, vec in zip(entries, vectors, strict=True):
                await conn.execute(
                    """
                    INSERT INTO agent.event_memory
                        (channel, channel_user_id, session_id,
                         content, kind, importance, keywords,
                         embedding, created_at, expires_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::timestamptz,
                            $9::timestamptz + ($10 || ' days')::interval)
                    """,
                    channel,
                    channel_user_id,
                    session_id,
                    entry.content,
                    entry.kind,
                    float(entry.importance),
                    entry.keywords,
                    vec,
                    entry.created_at,
                    str(ttl_days),
                )
                inserted += 1
    return inserted


async def insert_episodic_candidates(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
    session_id: str | None,
    candidates: list[MemoryCandidate],
    vectors: list[list[float]],
) -> int:
    """Append episodic candidates as ``tier='raw'`` rows (memory V2 write path).

    Unlike :func:`insert_event_memories`, this writes the V2 provenance columns
    (``subject``/``source``/``confidence``/``period``) and leaves ``expires_at``
    NULL — episodic forgetting is driven by monthly consolidation, not TTL.
    ``period`` is derived from each candidate's created-at month (``YYYY-MM``).

    Vectors are precomputed by the caller (the policy gate embeds once and
    reuses for both dedup and insert). Returns the count inserted.
    """
    if not candidates:
        return 0
    inserted = 0
    async with pool.acquire() as conn:
        async with conn.transaction():
            for cand, vec in zip(candidates, vectors, strict=True):
                now = datetime.now(UTC)
                period = now.strftime("%Y-%m")
                await conn.execute(
                    """
                    INSERT INTO agent.event_memory
                        (channel, channel_user_id, session_id, content, kind,
                         importance, keywords, embedding, created_at,
                         subject, source, confidence, tier, period)
                    VALUES ($1, $2, $3, $4, 'event', $5, $6, $7, $8::timestamptz,
                            $9, $10, $11, 'raw', $12)
                    """,
                    channel,
                    channel_user_id,
                    session_id,
                    cand.content,
                    float(cand.importance),
                    cand.keywords,
                    vec,
                    now,
                    cand.subject,
                    cand.source,
                    float(cand.confidence),
                    period,
                )
                inserted += 1
    return inserted


async def read_recent_event_memories(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Return the most recent non-expired event-memory rows for the identity.

    Result rows are plain dicts (not MemoryEntry) so the caller can
    decide whether to surface ``id`` / ``kind`` / ``importance`` to
    downstream consumers like the prompt renderer.

    Args:
        pool: asyncpg pool.
        channel: Channel slug.
        channel_user_id: External user id.
        limit: Cap on returned rows. ``0`` short-circuits.

    Returns:
        Newest-first list of ``{id, content, kind, importance, keywords,
        created_at}`` dicts. Empty list when nothing matches.
    """
    if limit <= 0:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, kind, importance, keywords, created_at
            FROM agent.event_memory
            WHERE channel = $1
              AND channel_user_id = $2
              AND (expires_at IS NULL OR expires_at > now())
            ORDER BY created_at DESC
            LIMIT $3
            """,
            channel,
            channel_user_id,
            limit,
        )
    return [
        {
            "id": str(r["id"]),
            "content": r["content"],
            "kind": r["kind"],
            "importance": float(r["importance"]),
            "keywords": list(r["keywords"] or []),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


__all__ = [
    "DEFAULT_EVENT_TTL_DAYS",
    "insert_event_memories",
    "read_recent_event_memories",
    # Re-export so callers (e.g., enter_node) keep working without
    # re-importing the prompts module separately.
    "render_recent_events_for_prompt",
]


# Re-export the renderer from the prompts module so existing imports
# (`from backend.v.memory.event_memory import render_recent_events_for_prompt`)
# keep working after the prompts centralization.
from backend.v.memory.prompts import render_recent_events_for_prompt  # noqa: E402

# Backwards-compat alias for ``datetime`` parameter typing in callers
# that previously imported it from this module.
__doc_extra_datetime = datetime
