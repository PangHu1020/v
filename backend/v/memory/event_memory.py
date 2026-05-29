"""Medium-term event memory (Phase-3 Group C, layer 2).

Reads against ``agent.session_memory`` filtered by the ``expires_at``
column added in scripts/sql/050_session_memory_ttl.sql. Each session-end
consolidation writes one row; rows live ``MEMORY_EVENT_TTL_DAYS`` days
(30 by default).

Two consumers:

- ``on_session_start`` injects the most-recent few entries into the
  system prompt for cross-session continuity ("上次我们聊到...")
- The mid-session compression node renders a summary line that replaces
  the compressed messages.

Writes happen in :mod:`backend.v.cron.tasks.consolidate_session`. This
module is read-only on purpose: keeping write SQL co-located with the
LLM-driven extraction it serves.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg


async def read_recent_event_memories(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Return the most recent non-expired event-memory rows for the identity.

    Joins ``agent.session_memory`` to ``agent.session`` so we can filter
    by ``(channel, channel_user_id)`` rather than per-session_id.

    Args:
        pool: asyncpg pool with codecs registered.
        channel: Channel slug.
        channel_user_id: External user id.
        limit: Cap on returned rows (chronological newest-first).

    Returns:
        A list of ``{summary, metadata, created_at}`` dicts, newest first.
        Empty list if the identity has no events or all have expired.
    """
    if limit <= 0:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT sm.summary, sm.metadata, sm.created_at
            FROM agent.session_memory sm
            JOIN agent.session s ON s.session_id = sm.session_id
            WHERE s.channel = $1
              AND s.channel_user_id = $2
              AND (sm.expires_at IS NULL OR sm.expires_at > now())
            ORDER BY sm.created_at DESC
            LIMIT $3
            """,
            channel,
            channel_user_id,
            limit,
        )
    out: list[dict[str, Any]] = []
    for row in rows:
        meta = row["metadata"]
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        out.append(
            {
                "summary": row["summary"],
                "metadata": dict(meta or {}),
                "created_at": row["created_at"],
            }
        )
    return out


# Re-export so existing callers (`from backend.v.memory.event_memory import
# render_recent_events_for_prompt`) continue to work after the prompts
# centralization.
from backend.v.memory.prompts import render_recent_events_for_prompt  # noqa: E402,F401
