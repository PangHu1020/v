"""Semantic-tier store: bi-temporal user memory (``agent.user_memory``).

The semantic tier holds overwritable customer attributes (preferences,
constraints, behavioural patterns). Unlike episodic events, an attribute has a
single *current truth* per ``attr_key``; a new value supersedes the old one
rather than coexisting with it.

Conflict model (the "three-step supersede"):

1. Read the active row for ``(channel, channel_user_id, attr_key)``.
2. If none → insert a new active row.
3. If one exists and the new candidate **wins arbitration**, close the old row
   (``status='superseded'``, ``valid_to=now()``, ``superseded_by=<new id>``)
   and insert the new active row in one transaction. The old row is never
   deleted — the chain is the audit log.

Arbitration (does the new value replace the active one?):
``stated`` beats ``inferred``; within the same source, higher ``confidence``
wins; ties go to the newer write (recency). An equal-value candidate is a
*reaffirmation* — we just refresh ``last_confirmed_at`` (and raise confidence
toward the stronger of the two), not a supersession.

After any write, :func:`rebuild_profile_cache` reprojects all active rows into
the ``agent.user_profile`` JSONB so the session-start injection stays a single
O(1) read.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import asyncpg

from backend.v.memory.types import MemoryCandidate, UserProfile
from backend.v.utils.logging import get_logger

_log = get_logger("memory.user_memory")

UpsertOutcome = Literal["inserted", "reaffirmed", "superseded", "skipped"]

# UserProfile fields that take a list value; everything else canonical is scalar.
_PROFILE_LIST_FIELDS = {"risk_flags"}
_PROFILE_CANONICAL = set(UserProfile.model_fields) - {"extras", "notes"}


def _new_wins(
    *,
    new_source: str,
    new_conf: float,
    old_source: str,
    old_conf: float,
) -> bool:
    """Arbitrate whether a differing new candidate replaces the active row.

    stated > inferred; then higher confidence; then recency (new wins on tie).
    """
    if new_source != old_source:
        return new_source == "stated"
    if new_conf != old_conf:
        return new_conf > old_conf
    return True  # same source + same confidence → newer write wins


async def read_active_user_memory(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
) -> list[dict[str, Any]]:
    """Return all current-truth (``status='active'``) rows for the identity."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, attr_key, attr_value, kind, source, confidence,
                   valid_from, last_confirmed_at
            FROM agent.user_memory
            WHERE channel = $1 AND channel_user_id = $2 AND status = 'active'
            ORDER BY attr_key
            """,
            channel,
            channel_user_id,
        )
    return [
        {
            "id": str(r["id"]),
            "attr_key": r["attr_key"],
            "attr_value": json.loads(r["attr_value"]) if r["attr_value"] else None,
            "kind": r["kind"],
            "source": r["source"],
            "confidence": float(r["confidence"]),
        }
        for r in rows
    ]


async def upsert_user_memory(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
    candidate: MemoryCandidate,
    session_id: str | None,
) -> UpsertOutcome:
    """Apply one semantic candidate via the three-step supersede protocol.

    ``candidate`` must carry ``attr_key`` / ``attr_value`` / ``kind`` (a semantic
    candidate). Returns which branch fired so the caller can log/aggregate.
    """
    if not candidate.attr_key or candidate.kind is None:
        return "skipped"

    new_value_json = json.dumps(candidate.attr_value, ensure_ascii=False)

    async with pool.acquire() as conn:
        async with conn.transaction():
            active = await conn.fetchrow(
                """
                SELECT id, attr_value, source, confidence
                FROM agent.user_memory
                WHERE channel = $1 AND channel_user_id = $2
                  AND attr_key = $3 AND status = 'active'
                FOR UPDATE
                """,
                channel,
                channel_user_id,
                candidate.attr_key,
            )

            if active is None:
                await _insert_active(
                    conn,
                    channel=channel,
                    channel_user_id=channel_user_id,
                    candidate=candidate,
                    value_json=new_value_json,
                    session_id=session_id,
                )
                return "inserted"

            old_value = json.loads(active["attr_value"]) if active["attr_value"] else None
            if old_value == candidate.attr_value:
                # Same value reaffirmed — refresh confirmation, lift confidence.
                await conn.execute(
                    """
                    UPDATE agent.user_memory
                    SET last_confirmed_at = now(),
                        confidence = GREATEST(confidence, $2),
                        source = CASE WHEN $3 = 'stated' THEN 'stated' ELSE source END
                    WHERE id = $1
                    """,
                    active["id"],
                    float(candidate.confidence),
                    candidate.source,
                )
                return "reaffirmed"

            if not _new_wins(
                new_source=candidate.source,
                new_conf=float(candidate.confidence),
                old_source=active["source"],
                old_conf=float(active["confidence"]),
            ):
                return "skipped"

            # Three-step supersede: close old, insert new active, link the chain.
            new_id = await _insert_active(
                conn,
                channel=channel,
                channel_user_id=channel_user_id,
                candidate=candidate,
                value_json=new_value_json,
                session_id=session_id,
            )
            await conn.execute(
                """
                UPDATE agent.user_memory
                SET status = 'superseded', valid_to = now(), superseded_by = $2
                WHERE id = $1
                """,
                active["id"],
                new_id,
            )
            return "superseded"


async def _insert_active(
    conn: asyncpg.Connection,
    *,
    channel: str,
    channel_user_id: str,
    candidate: MemoryCandidate,
    value_json: str,
    session_id: str | None,
) -> Any:
    """Insert a new active row, returning its id."""
    return await conn.fetchval(
        """
        INSERT INTO agent.user_memory
            (channel, channel_user_id, attr_key, attr_value, kind,
             source, confidence, status, valid_from, last_confirmed_at, session_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7, 'active', now(), now(), $8)
        RETURNING id
        """,
        channel,
        channel_user_id,
        candidate.attr_key,
        value_json,
        candidate.kind,
        candidate.source,
        float(candidate.confidence),
        session_id,
    )


def project_profile(active_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Project active user_memory rows into a ``UserProfile``-shaped dict.

    Canonical attr_keys map to top-level fields; everything else lands in
    ``extras`` (stringified). The result is what gets cached in
    ``agent.user_profile`` and injected at session start.
    """
    if not active_rows:
        return {}
    profile: dict[str, Any] = {}
    extras: dict[str, str] = {}
    for row in active_rows:
        key = row["attr_key"]
        value = row["attr_value"]
        if key in _PROFILE_CANONICAL:
            if key in _PROFILE_LIST_FIELDS:
                profile[key] = value if isinstance(value, list) else [str(value)]
            else:
                profile[key] = value
        else:
            extras[key] = str(value)
    if extras:
        profile["extras"] = extras
    # Validate-then-dump so the cache matches the renderer's expectations.
    # exclude_defaults drops the empty risk_flags/extras a bare model would add.
    try:
        return UserProfile.model_validate(profile).model_dump(
            mode="json", exclude_none=True, exclude_defaults=True
        )
    except Exception as exc:  # pragma: no cover — surfaces in logs
        _log.warning("memory.user_memory.profile_project_failed", error=type(exc).__name__)
        return profile


async def rebuild_profile_cache(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
) -> dict[str, Any]:
    """Reproject active rows into ``agent.user_profile`` and return the new cache."""
    active = await read_active_user_memory(pool, channel=channel, channel_user_id=channel_user_id)
    profile = project_profile(active)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO agent.user_profile (channel, channel_user_id, profile)
            VALUES ($1, $2, $3)
            ON CONFLICT (channel, channel_user_id) DO UPDATE
              SET profile = EXCLUDED.profile, updated_at = now()
            """,
            channel,
            channel_user_id,
            json.dumps(profile, ensure_ascii=False),
        )
    return profile
