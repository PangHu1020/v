"""Long-term memory reads (Phase-2 P3 / Phase-3 layer 3).

Owns just the ``agent.user_profile`` table — the truly-permanent layer.

Medium-term event memory (``agent.session_memory``) reads moved to
:mod:`backend.v.memory.event_memory` in Phase-3 Group C so each layer
of the temperature gradient lives in its own module.
"""

from __future__ import annotations

from typing import Any

import asyncpg


async def read_user_profile(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
) -> dict[str, Any] | None:
    """Return the stored profile JSON for the identity, or ``None`` if absent.

    Args:
        pool: asyncpg pool with codecs registered.
        channel: Channel slug.
        channel_user_id: External user id.

    Returns:
        The profile JSON as a dict, or ``None`` if no row exists.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT profile FROM agent.user_profile WHERE channel = $1 AND channel_user_id = $2",
            channel,
            channel_user_id,
        )
    if row is None:
        return None
    return dict(row["profile"]) if row["profile"] else {}
