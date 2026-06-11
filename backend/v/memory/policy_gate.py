"""Policy gate + episodic write helpers for the memory pipeline.

The write pipeline funnels every extracted candidate through a **policy gate**
before anything is persisted:

1. **Importance floor** — drop trivia below a configurable threshold.
2. **PII masking** — rewrite phone / id-card / address spans to masked tokens
   so durable storage never holds raw PII (see :mod:`backend.v.utils.pii`).
3. **Dedup** — episodic candidates that restate an existing same-subject row
   (high cosine similarity) are dropped; semantic dedup is handled by the
   bi-temporal upsert in :mod:`backend.v.memory.user_memory`, not here.

The gate is deterministic — no LLM call. Embeddings are computed once for the
surviving episodic candidates and reused for both the dedup check and the row
insert.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from backend.v.memory.types import MemoryCandidate
from backend.v.utils.logging import get_logger
from backend.v.utils.pii import mask_pii

_log = get_logger("memory.policy_gate")


def apply_importance_floor(
    candidates: list[MemoryCandidate], *, floor: float
) -> list[MemoryCandidate]:
    """Drop candidates whose importance is below ``floor``."""
    return [c for c in candidates if c.importance >= floor]


def apply_pii_mask(candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
    """Return copies with ``content`` (and string ``attr_value``) PII-masked.

    Masks the human-readable content for every candidate, and the value of
    semantic candidates when it is a string (numbers/lists pass through — a
    phone stored as ``attr_value`` would be a string and gets masked).
    """
    out: list[MemoryCandidate] = []
    for c in candidates:
        masked_value = mask_pii(c.attr_value) if isinstance(c.attr_value, str) else c.attr_value
        out.append(
            c.model_copy(update={"content": mask_pii(c.content), "attr_value": masked_value})
        )
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def dedup_episodic(
    pool: asyncpg.Pool,
    *,
    channel: str,
    channel_user_id: str,
    candidates: list[MemoryCandidate],
    vectors: list[list[float]],
    similarity_floor: float,
) -> tuple[list[MemoryCandidate], list[list[float]]]:
    """Drop episodic candidates that duplicate an existing same-subject row.

    For each candidate with a ``subject``, compare its vector against the
    embeddings of existing rows sharing that subject; if the best cosine is
    ``>= similarity_floor`` it's a restatement and is dropped. Candidates
    without a subject skip the check (kept). Returns the surviving candidates
    paired with their vectors, preserving order.
    """
    if not candidates:
        return [], []

    # Preload existing vectors per distinct subject in one pass.
    subjects = sorted({c.subject for c in candidates if c.subject})
    existing: dict[str, list[list[float]]] = {}
    if subjects:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT subject, embedding
                FROM agent.event_memory
                WHERE channel = $1 AND channel_user_id = $2
                  AND subject = ANY($3::text[])
                  AND (expires_at IS NULL OR expires_at > now())
                """,
                channel,
                channel_user_id,
                subjects,
            )
        for r in rows:
            existing.setdefault(r["subject"], []).append(_to_list(r["embedding"]))

    kept: list[MemoryCandidate] = []
    kept_vecs: list[list[float]] = []
    for cand, vec in zip(candidates, vectors, strict=True):
        if cand.subject and existing.get(cand.subject):
            best = max(_cosine(vec, ev) for ev in existing[cand.subject])
            if best >= similarity_floor:
                _log.info(
                    "memory.policy_gate.dedup_dropped", subject=cand.subject, sim=round(best, 3)
                )
                continue
        kept.append(cand)
        kept_vecs.append(vec)
    return kept, kept_vecs


def _to_list(embedding: Any) -> list[float]:
    """Coerce a pgvector value (str or sequence) to list[float]."""
    if isinstance(embedding, str):
        return [float(x) for x in embedding.strip("[]").split(",") if x]
    return [float(x) for x in embedding]
