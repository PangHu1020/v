"""Monthly episodic consolidation — the forgetting mechanism (memory V2).

Episodic memory is append-only, so without pruning it grows without bound. This
module bounds it by *consolidating* closed months: for each customer, raw events
in a month that has rolled over are grouped by ``subject``, each cluster is
summarised into one ``tier='summary'`` row, and then the low-value raws are
deleted — high-value ones survive ("用进废退", use-it-or-lose-it).

Trigger is **lazy**: the bus worker fires :func:`consolidate_user` fire-and-forget
when a customer starts a new session. No scheduler — work happens only for active
customers, which is exactly the population whose data is still growing. A churned
customer stops producing events, so their footprint is already bounded.

"Closed month" = any ``period`` (``YYYY-MM``) strictly before the current month.
A period is consolidatable when it still has un-consolidated raw rows
(``tier='raw' AND consolidated_into IS NULL``).

ACT-R base-level activation decides which raws survive::

    activation = w_imp·importance + w_acc·ln(1+access_count) − w_age·ln(1+age_days)

A frequently-recalled event (high ``access_count``) survives even at low
importance; a never-touched trivial one is dropped. Survivors are linked to the
summary (``consolidated_into``) so they are not re-summarised next month; they
remain ``tier='raw'`` and recallable.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any

import asyncpg
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage

from backend.v.memory.prompts import CONSOLIDATION_SUMMARY_SYSTEM_PROMPT
from backend.v.models.llm_caller import LLMCaller
from backend.v.utils.logging import bind_request, get_logger

_log = get_logger("memory.consolidation")

# Activation weights + survival threshold; stage 7 plumbs these from settings.
DEFAULT_W_IMPORTANCE = 1.0
DEFAULT_W_ACCESS = 0.5
DEFAULT_W_AGE = 0.3
DEFAULT_SURVIVAL_THRESHOLD = 0.5
# Don't bother summarising a 1-row cluster — nothing to compress.
DEFAULT_MIN_CLUSTER_SIZE = 2


def _current_period() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


def activation(
    *,
    importance: float,
    access_count: int,
    age_days: float,
    w_imp: float = DEFAULT_W_IMPORTANCE,
    w_acc: float = DEFAULT_W_ACCESS,
    w_age: float = DEFAULT_W_AGE,
) -> float:
    """ACT-R base-level activation for one episodic row."""
    return (
        w_imp * importance
        + w_acc * math.log1p(max(0, access_count))
        - w_age * math.log1p(max(0.0, age_days))
    )


async def _find_clusters(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
    current_period: str,
) -> list[tuple[str, str]]:
    """Return (period, subject) clusters in closed months awaiting consolidation."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT period, subject
            FROM agent.event_memory
            WHERE channel = $1 AND channel_user_id = $2
              AND tier = 'raw' AND consolidated_into IS NULL
              AND period IS NOT NULL AND subject IS NOT NULL
              AND period < $3
            GROUP BY period, subject
            """,
            channel,
            channel_user_id,
            current_period,
        )
    return [(r["period"], r["subject"]) for r in rows]


async def _fetch_cluster_rows(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
    period: str,
    subject: str,
) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, importance, access_count,
                   EXTRACT(EPOCH FROM (now() - created_at)) / 86400 AS age_days
            FROM agent.event_memory
            WHERE channel = $1 AND channel_user_id = $2
              AND period = $3 AND subject = $4
              AND tier = 'raw' AND consolidated_into IS NULL
            ORDER BY created_at
            """,
            channel,
            channel_user_id,
            period,
            subject,
        )
    return [
        {
            "id": r["id"],
            "content": r["content"],
            "importance": float(r["importance"]),
            "access_count": int(r["access_count"]),
            "age_days": float(r["age_days"]),
        }
        for r in rows
    ]


async def _summarise_cluster(
    llm_caller: LLMCaller,
    *,
    subject: str,
    period: str,
    rows: list[dict[str, Any]],
) -> tuple[str, float] | None:
    """LLM-compress a cluster into (summary_text, importance). None on failure."""
    listing = "\n".join(f"- {r['content']}" for r in rows)
    prompt = [
        SystemMessage(content=CONSOLIDATION_SUMMARY_SYSTEM_PROMPT),
        HumanMessage(
            content=f'<episodes subject="{subject}" period="{period}">\n{listing}\n</episodes>'
        ),
    ]
    try:
        result = await llm_caller.chat("memory_extract", prompt)
        payload = json.loads(result.message.content) if result.message.content else {}
        summary = str(payload.get("summary", "")).strip()
        if not summary:
            return None
        importance = float(payload.get("importance", max(r["importance"] for r in rows)))
        return summary, max(0.0, min(1.0, importance))
    except Exception as exc:
        _log.warning("memory.consolidation.summary_failed", error=type(exc).__name__)
        return None


async def consolidate_user(
    ctx: dict[str, Any],
    *,
    channel: str,
    channel_user_id: str,
) -> dict[str, int] | None:
    """Consolidate one customer's closed-month episodic clusters (lazy trigger).

    Pulls ``pool`` / ``llm_caller`` / ``embedder`` (and optional ``settings``)
    from ``ctx``. For each (period, subject) cluster of un-consolidated raws in a
    closed month: summarise → insert a ``tier='summary'`` row → delete low-
    activation raws and link the survivors to the summary.

    Returns ``{"summaries": S, "raws_deleted": D, "raws_kept": K}`` or ``None``
    when there's nothing to do.
    """
    pool: asyncpg.Pool = ctx["pool"]
    llm_caller: LLMCaller = ctx["llm_caller"]
    embedder: Embeddings = ctx["embedder"]
    w_imp, w_acc, w_age, threshold, min_cluster = _consolidation_params(ctx.get("settings"))

    current = _current_period()
    with bind_request(channel=channel, channel_user_id=channel_user_id):
        clusters = await _find_clusters(
            pool,
            channel=channel,
            channel_user_id=channel_user_id,
            current_period=current,
        )
        if not clusters:
            return None

        summaries = raws_deleted = raws_kept = 0
        for period, subject in clusters:
            rows = await _fetch_cluster_rows(
                pool,
                channel=channel,
                channel_user_id=channel_user_id,
                period=period,
                subject=subject,
            )
            if len(rows) < min_cluster:
                continue

            summarised = await _summarise_cluster(
                llm_caller, subject=subject, period=period, rows=rows
            )
            if summarised is None:
                continue
            summary_text, summary_importance = summarised
            vector = await embedder.aembed_query(summary_text)

            summary_id = await _insert_summary(
                pool,
                channel=channel,
                channel_user_id=channel_user_id,
                subject=subject,
                period=period,
                content=summary_text,
                importance=summary_importance,
                vector=vector,
            )
            summaries += 1

            deleted, kept = await _prune_cluster(
                pool,
                rows=rows,
                summary_id=summary_id,
                w_imp=w_imp,
                w_acc=w_acc,
                w_age=w_age,
                threshold=threshold,
            )
            raws_deleted += deleted
            raws_kept += kept

        if summaries == 0:
            return None
        _log.info(
            "memory.consolidation.done",
            summaries=summaries,
            raws_deleted=raws_deleted,
            raws_kept=raws_kept,
        )
        return {"summaries": summaries, "raws_deleted": raws_deleted, "raws_kept": raws_kept}


async def _insert_summary(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
    subject: str,
    period: str,
    content: str,
    importance: float,
    vector: list[float],
) -> Any:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO agent.event_memory
                (channel, channel_user_id, content, kind, importance, keywords,
                 embedding, subject, tier, period, source, confidence)
            VALUES ($1, $2, $3, 'event', $4, '{}', $5, $6, 'summary', $7, 'inferred', 1.0)
            RETURNING id
            """,
            channel,
            channel_user_id,
            content,
            float(importance),
            vector,
            subject,
            period,
        )


async def _prune_cluster(
    pool: asyncpg.Pool,
    *,
    rows: list[dict[str, Any]],
    summary_id: Any,
    w_imp: float,
    w_acc: float,
    w_age: float,
    threshold: float,
) -> tuple[int, int]:
    """Delete low-activation raws; link survivors to the summary. Returns (del, kept)."""
    to_delete: list[Any] = []
    to_keep: list[Any] = []
    for r in rows:
        act = activation(
            importance=r["importance"],
            access_count=r["access_count"],
            age_days=r["age_days"],
            w_imp=w_imp,
            w_acc=w_acc,
            w_age=w_age,
        )
        (to_keep if act >= threshold else to_delete).append(r["id"])

    async with pool.acquire() as conn:
        async with conn.transaction():
            if to_delete:
                await conn.execute(
                    "DELETE FROM agent.event_memory WHERE id = ANY($1::uuid[])",
                    to_delete,
                )
            if to_keep:
                # Link survivors so they're excluded from next month's scan.
                await conn.execute(
                    """
                    UPDATE agent.event_memory
                    SET consolidated_into = $2
                    WHERE id = ANY($1::uuid[])
                    """,
                    to_keep,
                    summary_id,
                )
    return len(to_delete), len(to_keep)


def _consolidation_params(settings: Any) -> tuple[float, float, float, float, int]:
    """Resolve (w_imp, w_acc, w_age, survival_threshold, min_cluster) from settings."""
    mem = getattr(settings, "memory", None)
    return (
        float(getattr(mem, "activation_w_importance", DEFAULT_W_IMPORTANCE)),
        float(getattr(mem, "activation_w_access", DEFAULT_W_ACCESS)),
        float(getattr(mem, "activation_w_age", DEFAULT_W_AGE)),
        float(getattr(mem, "consolidation_survival_threshold", DEFAULT_SURVIVAL_THRESHOLD)),
        int(getattr(mem, "consolidation_min_cluster_size", DEFAULT_MIN_CLUSTER_SIZE)),
    )
