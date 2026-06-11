"""Session-end memory consolidation: working memory / transcript → durable stores.

Triggered fire-and-forget when a session expires. Runs ONE LLM extraction over
the session's distilled signal and routes the result into the two durable tiers:

- **user_candidates** → ``agent.user_memory`` via the bi-temporal upsert
  (:mod:`backend.v.memory.user_memory`), then the ``user_profile`` JSONB cache
  is rebuilt from the active rows.
- **episodic_candidates** → ``agent.event_memory`` as ``tier='raw'`` rows,
  after the policy gate masks PII, drops trivia, and de-duplicates against
  existing same-subject events.

The single-call design is deliberate: an earlier version extracted twice (once
to populate working memory, once to promote), burning tokens. Here the input is
whichever distilled signal exists — the session's working memory if mid-session
compression produced any, otherwise the raw transcript for a short session —
and exactly one extraction call fans out to both stores.

What this module is NOT responsible for: mid-session compression (owned by
``compression_node``) and monthly consolidation/forgetting (owned by
``backend.v.memory.consolidation``).
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from backend.v.memory.event_memory import insert_episodic_candidates
from backend.v.memory.policy_gate import apply_importance_floor, apply_pii_mask, dedup_episodic
from backend.v.memory.prompts import SESSION_END_EXTRACTION_SYSTEM_PROMPT
from backend.v.memory.types import MemoryEntry, MemoryExtraction
from backend.v.memory.user_memory import (
    read_active_user_memory,
    rebuild_profile_cache,
    upsert_user_memory,
)
from backend.v.memory.working import delete_working_memory, read_working_memory
from backend.v.models.llm_caller import LLMCaller
from backend.v.utils.logging import bind_request, get_logger

_log = get_logger("memory.session_end")

# Policy-gate defaults; stage 7 plumbs these from MemorySettings via ctx.
DEFAULT_IMPORTANCE_FLOOR = 0.3
DEFAULT_DEDUP_SIMILARITY_FLOOR = 0.92


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


def _render_working(entries: list[MemoryEntry]) -> str:
    """Render distilled working-memory entries as the extraction input."""
    return "\n".join(f"- [{e.kind}] {e.content}" for e in entries)


def _render_transcript(messages: list[BaseMessage]) -> str:
    """Render a raw transcript (short-session fallback) as the extraction input."""
    lines: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"客户：{m.content}")
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            lines.append(f"助理：{m.content}")
    return "\n".join(lines)


def _build_prompt(
    existing_user_memory: list[dict[str, Any]], session_input: str
) -> list[BaseMessage]:
    existing_json = json.dumps(
        [
            {"attr_key": r["attr_key"], "attr_value": r["attr_value"], "kind": r["kind"]}
            for r in existing_user_memory
        ],
        ensure_ascii=False,
    )
    return [
        SystemMessage(content=SESSION_END_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"<existing_user_memory>\n{existing_json}\n</existing_user_memory>\n\n"
                f"<session_input>\n{session_input}\n</session_input>"
            )
        ),
    ]


async def promote_to_long_term(
    ctx: dict[str, Any],
    *,
    session_id: str,
) -> dict[str, int] | None:
    """Consolidate an expired session into user + episodic memory (V2 pipeline).

    Pulls its inputs from ``ctx``: ``pool``, ``redis``, ``llm_caller``,
    ``embedder``, optional ``fallback_messages`` (raw transcript for a short
    session whose working memory is empty), and optional ``settings`` (for
    policy-gate thresholds).

    Returns ``{"user_memory_written": N, "events_inserted": M}`` on success, or
    ``None`` when there's no session row / nothing to consolidate / the LLM call
    fails. On success the working-memory list is deleted so a re-run no-ops.
    """
    pool: asyncpg.Pool = ctx["pool"]
    redis = ctx["redis"]
    llm_caller: LLMCaller = ctx["llm_caller"]
    embedder: Embeddings = ctx["embedder"]
    fallback_messages: list[BaseMessage] = ctx.get("fallback_messages") or []
    floor, dedup_floor = _gate_thresholds(ctx.get("settings"))

    with bind_request(session_id=session_id):
        identity = await _read_session_identity(pool, session_id)
        if identity is None:
            _log.info("memory.session_end.no_session_row")
            return None
        channel, channel_user_id = identity

        working = await read_working_memory(redis, session_id=session_id)
        if working:
            session_input = _render_working(working)
        elif fallback_messages:
            session_input = _render_transcript(fallback_messages)
        else:
            _log.info("memory.session_end.no_input")
            return None
        if not session_input.strip():
            return None

        existing = await read_active_user_memory(
            pool, channel=channel, channel_user_id=channel_user_id
        )

        prompt = _build_prompt(existing, session_input)
        try:
            result = await llm_caller.chat("memory_extract", prompt, structured=MemoryExtraction)
        except Exception as exc:
            _log.error("memory.session_end.llm_failed", error=type(exc).__name__)
            return None

        extraction = result.parsed
        if not isinstance(extraction, MemoryExtraction):
            _log.error("memory.session_end.parse_failed")
            return None

        # --- Policy gate (deterministic): importance floor + PII mask ---
        user_c = apply_pii_mask(apply_importance_floor(extraction.user_candidates, floor=floor))
        epi_c = apply_pii_mask(apply_importance_floor(extraction.episodic_candidates, floor=floor))

        # --- Semantic tier: bi-temporal upsert + profile-cache rebuild ---
        user_written = 0
        for cand in user_c:
            outcome = await upsert_user_memory(
                pool,
                channel=channel,
                channel_user_id=channel_user_id,
                candidate=cand,
                session_id=session_id,
            )
            if outcome in ("inserted", "superseded", "reaffirmed"):
                user_written += 1
        if user_written:
            await rebuild_profile_cache(pool, channel=channel, channel_user_id=channel_user_id)

        # --- Episodic tier: embed once → dedup → append ---
        events_inserted = 0
        if epi_c and embedder:
            vectors = await embedder.aembed_documents([c.content for c in epi_c])
            kept, kept_vecs = await dedup_episodic(
                pool,
                channel=channel,
                channel_user_id=channel_user_id,
                candidates=epi_c,
                vectors=vectors,
                similarity_floor=dedup_floor,
            )
            events_inserted = await insert_episodic_candidates(
                pool,
                channel=channel,
                channel_user_id=channel_user_id,
                session_id=session_id,
                candidates=kept,
                vectors=kept_vecs,
            )

        await delete_working_memory(redis, session_id=session_id)

        _log.info(
            "memory.session_end.done",
            user_memory_written=user_written,
            events_inserted=events_inserted,
        )
        return {"user_memory_written": user_written, "events_inserted": events_inserted}


def _gate_thresholds(settings: Any) -> tuple[float, float]:
    """Resolve (importance_floor, dedup_similarity_floor) from settings or defaults."""
    mem = getattr(settings, "memory", None)
    floor = getattr(mem, "consolidation_importance_floor", DEFAULT_IMPORTANCE_FLOOR)
    dedup = getattr(mem, "consolidation_dedup_similarity_floor", DEFAULT_DEDUP_SIMILARITY_FLOOR)
    return float(floor), float(dedup)
