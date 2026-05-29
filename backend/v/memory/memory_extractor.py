"""Long-term memory promotion (Phase-3 reshape).

Triggered at session end. Reads the active session's working memory +
the existing user_profile, asks the LLM to:

1. Decide which entries from working memory have **stable, long-term
   value** about THIS customer (preferred courier, allergy, preferred
   salutation) — those become a partial :class:`UserProfile` patch
   merged into ``agent.user_profile``.
2. Decide which entries (or new sentences synthesized from the
   transcript) deserve a **30-day persistent footprint** in
   ``agent.event_memory`` for cross-session recall.

What this module is NOT responsible for:

- Mid-session compression — owned by ``compression_node``.
- Per-turn working-memory writes — owned by the consolidator
  (``consolidate_session``) which calls in here at session end only.

The result is :class:`backend.v.memory.types.ExtractionResult`:
``profile_updates`` + ``event_memories``. Working memory is dropped on
success because the long-term footprint has already been computed.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from backend.v.memory.event_memory import insert_event_memories
from backend.v.memory.prompts import LONG_TERM_PROMOTION_SYSTEM_PROMPT
from backend.v.memory.types import ExtractionResult, MemoryEntry, UserProfile
from backend.v.memory.working import delete_working_memory, read_working_memory
from backend.v.models.llm_caller import LLMCaller
from backend.v.utils.logging import bind_request, get_logger

_log = get_logger("memory.long_term_promotion")


def _build_prompt(
    existing_profile: dict[str, Any],
    working_entries: list[MemoryEntry],
) -> list[BaseMessage]:
    """Assemble the SystemMessage + HumanMessage for the LLM extractor."""
    profile_json = json.dumps(existing_profile, ensure_ascii=False, indent=2)
    working_json = json.dumps(
        [e.model_dump(mode="json") for e in working_entries],
        ensure_ascii=False,
        indent=2,
    )
    return [
        SystemMessage(content=LONG_TERM_PROMOTION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"<existing_profile>\n{profile_json}\n</existing_profile>\n\n"
                f"<working_memory>\n{working_json}\n</working_memory>"
            )
        ),
    ]


async def _read_session_identity(
    pool: asyncpg.Pool,
    session_id: str,
) -> tuple[str, str] | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT channel, channel_user_id FROM agent.session WHERE session_id = $1",
            session_id,
        )
    if row is None:
        return None
    return row["channel"], row["channel_user_id"]


async def _read_existing_profile(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT profile FROM agent.user_profile WHERE channel = $1 AND channel_user_id = $2",
            channel,
            channel_user_id,
        )
    if row is None or row["profile"] is None:
        return {}
    return dict(row["profile"])


def _merge_profile(existing: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Merge ``updates`` into ``existing`` with shallow + ``extras`` deep-merge.

    The model returns a partial ``UserProfile`` shape. We drop unknown
    top-level keys (they go in ``extras`` instead) but keep the
    customer's existing fields when ``updates`` doesn't mention them.
    """
    canonical_keys = set(UserProfile.model_fields) - {"extras", "notes"}
    merged = dict(existing)

    for key, value in updates.items():
        if key in canonical_keys:
            # Recency wins: an explicit update overwrites; an explicit
            # ``None`` clears.
            merged[key] = value
        elif key == "extras" and isinstance(value, dict):
            existing_extras = dict(merged.get("extras") or {})
            for k, v in value.items():
                existing_extras[k] = v
            merged["extras"] = existing_extras
        elif key == "notes":
            merged["notes"] = value
        else:
            # Unknown top-level key from the LLM → stash in extras so we
            # don't drop information; preserves the "open extension" idea.
            extras = dict(merged.get("extras") or {})
            extras[str(key)] = str(value)
            merged["extras"] = extras

    # Validate-then-dump so we keep stable field ordering and reject
    # malformed shapes early. Tolerant of partial profiles.
    try:
        return UserProfile.model_validate(merged).model_dump(mode="json", exclude_none=True)
    except Exception as exc:  # pragma: no cover — surfaces in logs
        _log.warning("memory.long_term.profile_validate_failed", error=type(exc).__name__)
        return merged


async def _upsert_user_profile(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
    profile: dict[str, Any],
) -> None:
    if not profile:
        return
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
            profile,
        )


async def promote_to_long_term(
    ctx: dict[str, Any],
    *,
    session_id: str,
) -> dict[str, int] | None:
    """Promote this session's working memory to long-term storage.

    Returns ``{"profile_updated": 0|1, "events_inserted": N}`` on
    success or ``None`` when there's nothing to promote.

    On success, the session's working-memory list is deleted so a
    re-run is a clean no-op.
    """
    pool: asyncpg.Pool = ctx["pool"]
    redis = ctx["redis"]
    llm_caller: LLMCaller = ctx["llm_caller"]
    embedder: Embeddings = ctx["embedder"]

    with bind_request(session_id=session_id):
        identity = await _read_session_identity(pool, session_id)
        if identity is None:
            _log.info("memory.long_term.no_session_row")
            return None
        channel, channel_user_id = identity

        working = await read_working_memory(redis, session_id=session_id)
        if not working:
            _log.info("memory.long_term.no_working_memory")
            return None

        existing = await _read_existing_profile(
            pool, channel=channel, channel_user_id=channel_user_id
        )

        prompt = _build_prompt(existing, working)
        try:
            result = await llm_caller.chat("memory_extract", prompt, structured=ExtractionResult)
        except Exception as exc:
            _log.error("memory.long_term.llm_failed", error=type(exc).__name__)
            return None

        try:
            payload = (
                json.loads(result.message.content)
                if isinstance(result.message.content, str)
                else {}
            )
            extracted = ExtractionResult.model_validate(payload)
        except Exception as exc:
            _log.error("memory.long_term.parse_failed", error=type(exc).__name__)
            return None

        profile_updated = 0
        if extracted.profile_updates:
            merged = _merge_profile(existing, extracted.profile_updates)
            if merged != existing:
                await _upsert_user_profile(
                    pool,
                    channel=channel,
                    channel_user_id=channel_user_id,
                    profile=merged,
                )
                profile_updated = 1

        events_inserted = await insert_event_memories(
            pool,
            channel=channel,
            channel_user_id=channel_user_id,
            session_id=session_id,
            entries=extracted.event_memories,
            embedder=embedder,
        )

        # Working memory has served its purpose for this session.
        await delete_working_memory(redis, session_id=session_id)

        _log.info(
            "memory.long_term.done",
            profile_updated=profile_updated,
            events_inserted=events_inserted,
        )
        return {"profile_updated": profile_updated, "events_inserted": events_inserted}


# Public alias kept for ARQ workers / other callers that import the
# Phase-2 name. The function shape is unchanged but the body is the
# Phase-3 promotion above.
extract_session_memory = promote_to_long_term
