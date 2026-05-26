"""Working-memory snapshots in Redis.

The on-session-start hook caches the user's long-term ``user_profile`` in
Redis so subsequent turns within the same 30-minute session don't re-hit
Postgres. The Redis-backed checkpointer (see ``checkpointer.py``) handles
LangGraph state itself; this module is just the auxiliary user-profile
snapshot.
"""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as redis_async


def _profile_key(channel: str, channel_user_id: str) -> str:
    return f"profile:{channel}:{channel_user_id}"


async def cache_user_profile(
    client: redis_async.Redis,
    *,
    channel: str,
    channel_user_id: str,
    profile: dict[str, Any],
    ttl_seconds: int,
) -> None:
    """Store the user's profile JSON for the working-memory window.

    Args:
        client: Redis async client.
        channel: Channel slug.
        channel_user_id: External user id.
        profile: JSON-serializable profile dict.
        ttl_seconds: TTL for the cached snapshot.
    """
    key = _profile_key(channel, channel_user_id)
    await client.set(key, json.dumps(profile, ensure_ascii=False), ex=ttl_seconds)


async def get_cached_user_profile(
    client: redis_async.Redis,
    *,
    channel: str,
    channel_user_id: str,
) -> dict[str, Any] | None:
    """Return the cached profile if present, else ``None``."""
    key = _profile_key(channel, channel_user_id)
    raw = await client.get(key)
    if raw is None:
        return None
    return json.loads(raw)
