"""Long-term memory reads (Phase-1).

Only the read path against ``agent.user_profile`` lands in Phase-1 to keep
the on-session-start hook honest. The write path (driven by the
``memory_extractor`` background analyst) is Phase-2.
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
