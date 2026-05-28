"""Session-lifecycle hook: bootstrap context for the next turn.

Runs out-of-band before the bus worker invokes the LangGraph turn so
the graph receives ``state["user_profile"]`` already populated AND a
list of recent event-memory summaries for continuity across sessions.

Phase-3 expanded this hook from "user profile only" to a two-thing
returner: long-term profile + the last N non-expired events from
``agent.session_memory``. The values are surfaced separately so
``enter_node`` can render the system prompt correctly while the graph
state still keeps ``user_profile`` typed as a dict.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import redis.asyncio as redis_async

from backend.v.memory.event_memory import read_recent_event_memories
from backend.v.memory.long_term import read_user_profile
from backend.v.memory.working import cache_user_profile, get_cached_user_profile
from backend.v.utils.logging import get_logger

_log = get_logger("hooks.session")


async def on_session_start(
    *,
    pool: asyncpg.Pool,
    redis: redis_async.Redis,
    channel: str,
    channel_user_id: str,
    cache_ttl_seconds: int,
    recent_events_limit: int = 0,
) -> dict[str, Any]:
    """Return the bootstrap dict for the next turn.

    Result shape::

        {
            "profile": {...},            # always a dict (possibly empty)
            "recent_events": [           # newest first; possibly empty
                {"summary": "...", "metadata": {...}, "created_at": ...},
                ...
            ],
        }

    Profile is cached in Redis (TTL = ``cache_ttl_seconds``) so the
    common case skips the Postgres round-trip. Recent events are NOT
    cached; they're cheap to query and can change between turns
    (consolidate_session writes here while the session is still alive).

    Args:
        pool: Postgres pool with codecs registered.
        redis: Redis client.
        channel: Channel slug.
        channel_user_id: External user id.
        cache_ttl_seconds: TTL for the working-memory profile cache
            after a fresh fetch.
        recent_events_limit: How many medium-term event-memory rows to
            return. ``0`` disables the lookup entirely.
    """
    cached = await get_cached_user_profile(redis, channel=channel, channel_user_id=channel_user_id)
    if cached is None:
        profile = await read_user_profile(pool, channel=channel, channel_user_id=channel_user_id)
        if profile is None:
            profile = {}
        await cache_user_profile(
            redis,
            channel=channel,
            channel_user_id=channel_user_id,
            profile=profile,
            ttl_seconds=cache_ttl_seconds,
        )
        _log.debug("on_session_start.cache_miss_warmed", has_profile=bool(profile))
    else:
        profile = cached
        _log.debug("on_session_start.cache_hit")

    recent_events: list[dict[str, Any]] = []
    if recent_events_limit > 0:
        recent_events = await read_recent_event_memories(
            pool,
            channel=channel,
            channel_user_id=channel_user_id,
            limit=recent_events_limit,
        )
        _log.debug("on_session_start.recent_events_loaded", count=len(recent_events))

    return {"profile": profile, "recent_events": recent_events}
