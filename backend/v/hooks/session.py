"""Session-lifecycle hook: load user profile on session start.

Runs out-of-band before the bus worker invokes the LangGraph turn so the
graph receives ``state["user_profile"]`` already populated. Reads the
working-memory cache first; falls back to the Postgres long-term store and
warms the cache on miss.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import redis.asyncio as redis_async

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
) -> dict[str, Any]:
    """Return the user_profile dict for this identity (cached or fetched).

    Args:
        pool: Postgres pool with codecs registered.
        redis: Redis client.
        channel: Channel slug.
        channel_user_id: External user id.
        cache_ttl_seconds: TTL for the working-memory cache after a fresh fetch.

    Returns:
        The profile dict (possibly empty). Never returns ``None`` so the
        graph can rely on ``state["user_profile"]`` being a dict.
    """
    cached = await get_cached_user_profile(redis, channel=channel, channel_user_id=channel_user_id)
    if cached is not None:
        _log.debug("on_session_start.cache_hit")
        return cached

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
    return profile
