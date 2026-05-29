"""Working memory: Redis snapshot for the active session.

Two things live here:

- The user_profile cache. ``on_session_start`` reads the long-term
  ``agent.user_profile`` row once and pins it into Redis with the
  working-memory TTL so subsequent turns within the same session skip
  the Postgres round-trip.
- The active session's working memory itself: a list of
  :class:`backend.v.memory.types.MemoryEntry`. Append-only during a
  session, evaporates with the session TTL.

Both are scoped to the same lifetime (TTL = working memory window) and
disappear together when the session ages out.
"""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as redis_async

from backend.v.memory.types import MemoryEntry


def _profile_key(channel: str, channel_user_id: str) -> str:
    return f"profile:{channel}:{channel_user_id}"


def _working_memory_key(session_id: str) -> str:
    return f"working_memory:{session_id}"


# ── User-profile cache ────────────────────────────────────────────────────────


async def cache_user_profile(
    client: redis_async.Redis,
    *,
    channel: str,
    channel_user_id: str,
    profile: dict[str, Any],
    ttl_seconds: int,
) -> None:
    """Store the customer's profile JSON for the working-memory window."""
    key = _profile_key(channel, channel_user_id)
    await client.set(key, json.dumps(profile, ensure_ascii=False), ex=ttl_seconds)


async def get_cached_user_profile(
    client: redis_async.Redis,
    *,
    channel: str,
    channel_user_id: str,
) -> dict[str, Any] | None:
    """Return the cached profile, else ``None``."""
    key = _profile_key(channel, channel_user_id)
    raw = await client.get(key)
    if raw is None:
        return None
    return json.loads(raw)


# ── Working memory entries ────────────────────────────────────────────────────


async def append_working_memory(
    client: redis_async.Redis,
    *,
    session_id: str,
    entries: list[MemoryEntry],
    ttl_seconds: int,
) -> None:
    """Push ``entries`` onto the working-memory list and refresh TTL.

    Stored as a Redis list of JSON-encoded MemoryEntry payloads. The
    list is naturally ordered oldest-first; readers reverse for
    newest-first injection.

    Empty ``entries`` is a no-op (we deliberately don't bump the TTL on
    a zero-write turn — the session checkpointer handles that).
    """
    if not entries:
        return
    key = _working_memory_key(session_id)
    payloads = [e.model_dump_json() for e in entries]
    pipe = client.pipeline()
    pipe.rpush(key, *payloads)
    pipe.expire(key, ttl_seconds)
    await pipe.execute()


async def read_working_memory(
    client: redis_async.Redis,
    *,
    session_id: str,
) -> list[MemoryEntry]:
    """Return all working-memory entries for the session, oldest-first.

    Empty list if the key is missing or every entry fails to parse.
    """
    key = _working_memory_key(session_id)
    raw_list = await client.lrange(key, 0, -1)
    out: list[MemoryEntry] = []
    for raw in raw_list:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            out.append(MemoryEntry.model_validate_json(raw))
        except Exception:  # noqa: S112 — drop legacy/malformed entries
            continue
    return out


async def delete_working_memory(
    client: redis_async.Redis,
    *,
    session_id: str,
) -> None:
    """Drop the session's working-memory list. Used at consolidation."""
    await client.delete(_working_memory_key(session_id))
