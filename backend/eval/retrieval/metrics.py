"""Retrieval metrics for the RAG eval harness.

All functions are pure: they take a ranked list of retrieved ``source_id``
strings plus the set of gold ``source_id`` strings and return a float. No IO,
no LLM, no Milvus — so they are cheap and deterministic to unit-test.

Conventions:

- ``retrieved`` is ordered best-first (rank 1 = index 0).
- ``gold`` is the set of relevant source ids for the query (usually size 1).
- ``k`` truncates ``retrieved`` to its top-k before scoring.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def hit_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """1.0 if any gold id appears in the top-k, else 0.0."""
    gold_set = set(gold)
    return 1.0 if any(r in gold_set for r in retrieved[:k]) else 0.0


def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Fraction of gold ids found in the top-k.

    With a single gold id this is equivalent to hit@k.
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    found = sum(1 for g in gold_set if g in set(retrieved[:k]))
    return found / len(gold_set)


def mrr_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Reciprocal rank of the first gold id within the top-k (0.0 if none)."""
    gold_set = set(gold)
    for i, r in enumerate(retrieved[:k], start=1):
        if r in gold_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k.

    DCG uses ``1/log2(rank+1)`` gain for each gold hit; IDCG is the best
    achievable ordering given ``min(len(gold), k)`` relevant docs.
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    dcg = 0.0
    for i, r in enumerate(retrieved[:k], start=1):
        if r in gold_set:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(
    rows: list[dict],
    ks: Sequence[int],
) -> dict[str, float]:
    """Average each metric across many queries.

    Args:
        rows: list of ``{"retrieved": [...], "gold": [...]}`` dicts.
        ks: cutoffs to compute (e.g. ``[1, 3, 5]``).

    Returns:
        Flat dict like ``{"hit@1": .., "recall@5": .., "mrr@5": .., "ndcg@5": ..}``.
    """
    if not rows:
        return {}
    out: dict[str, float] = {}
    n = len(rows)
    for k in ks:
        out[f"hit@{k}"] = sum(hit_at_k(r["retrieved"], r["gold"], k) for r in rows) / n
        out[f"recall@{k}"] = sum(recall_at_k(r["retrieved"], r["gold"], k) for r in rows) / n
        out[f"mrr@{k}"] = sum(mrr_at_k(r["retrieved"], r["gold"], k) for r in rows) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r["retrieved"], r["gold"], k) for r in rows) / n
    return out
