"""Short-term session memory (Phase-3 三层温度梯度第一层).

The 会话记忆 layer holds preferences and observations extracted from the
current session's messages. Lives in Redis with the same TTL as the
working-memory checkpointer so it disappears together with the session
itself.

Two write paths feed it:

- :func:`backend.v.cron.tasks.consolidate_session.consolidate_session`
  runs at TTL-Δ or when the compression threshold is breached and writes
  the structured findings produced by the LLM.
- A future "incremental observer" hook may append to ``observations``
  on each turn; not implemented in this commit.

One read path consumes it:

- The session-end promotion in
  :mod:`backend.v.memory.memory_extractor` reads the full record to
  decide which preferences cross over into the long-term ``user_profile``.

The mid-session compression node also reads ``preferences`` and
re-injects them as a SystemMessage so the post-compression LLM call
keeps personalization context.
"""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as redis_async


def _key(session_id: str) -> str:
    return f"session_memory:{session_id}"


async def write_session_memory(
    redis: redis_async.Redis,
    *,
    session_id: str,
    preferences: dict[str, Any],
    observations: list[str],
    ttl_seconds: int,
) -> None:
    """Overwrite the short-term record. Idempotent on repeat writes."""
    payload = {
        "preferences": preferences,
        "observations": observations,
    }
    await redis.set(_key(session_id), json.dumps(payload, ensure_ascii=False), ex=ttl_seconds)


async def read_session_memory(
    redis: redis_async.Redis,
    *,
    session_id: str,
) -> dict[str, Any] | None:
    """Return ``{preferences, observations}`` or ``None`` if missing."""
    raw = await redis.get(_key(session_id))
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def delete_session_memory(
    redis: redis_async.Redis,
    *,
    session_id: str,
) -> None:
    """Drop the record. Used after session-end promotion succeeds."""
    await redis.delete(_key(session_id))


def render_for_prompt(record: dict[str, Any] | None) -> str:
    """Render the record as a compact Chinese block for system-prompt injection."""
    if not record:
        return ""
    prefs = record.get("preferences") or {}
    obs = record.get("observations") or []
    lines: list[str] = []
    if prefs:
        lines.append("本次会话已知偏好：")
        for k, v in prefs.items():
            lines.append(f"- {k}：{v}")
    if obs:
        lines.append("本次会话观察：")
        for item in obs:
            lines.append(f"- {item}")
    return "\n".join(lines)
