"""Confidence gate for the cascade retriever.

The cascade decides at each stage whether the current results are *good enough*
to return, or whether to escalate to the next (more expensive) stage. The old
design used a single absolute score floor per stage — brittle because:

- Absolute score height is not relevance: some query phrasings push every score
  up, others push everything down. A global cutoff treats those identically.
- Stage-2's ``WeightedRanker`` output is an unnormalized fused-rank score on an
  arbitrary scale, so an absolute floor there has no stable meaning.
- The numbers are tied to one embedding model + corpus; re-seeding or swapping
  the model silently drifts them wrong.

This module replaces that with a **relative-margin gate**: a stage exits when
its top hit is *confidently separated* from the runner-up, with only a low
absolute floor as a sanity check. The margin is computed relative to the top
score, so it is scale-invariant — the same params work whether scores live in
[0.6, 0.95] (dense cosine) or [0.4, 0.9] (hybrid fused rank).

All functions are pure (no IO, no LLM), so the gate is cheap to unit-test and
the parameters can be calibrated offline from the eval set.
"""

from __future__ import annotations

from dataclasses import dataclass

_EPS = 1e-9


@dataclass(frozen=True)
class GateParams:
    """One stage's exit gate.

    Attributes:
        floor: Absolute minimum the top-1 score must clear (sanity check).
            Low by design — the real decision is the margin.
        rel_margin: Minimum *relative* separation ``(top1 - top2) / top1``.
            0.0 disables the margin check (floor-only behaviour). A single
            result (no runner-up) is treated as maximally separated.
        min_k: Minimum number of results required to consider exiting.
    """

    floor: float
    rel_margin: float
    min_k: int = 1


def confidence_signals(scores: list[float]) -> dict[str, float]:
    """Compute scale-invariant confidence signals from a result score list.

    Returns ``top1``, ``top2``, ``margin`` (absolute top1-top2), ``rel_margin``
    ((top1-top2)/top1), and ``mean``. Empty input yields all-zeros. Used by the
    gate and surfaced in traces / calibration.
    """
    if not scores:
        return {"top1": 0.0, "top2": 0.0, "margin": 0.0, "rel_margin": 0.0, "mean": 0.0}
    ordered = sorted(scores, reverse=True)
    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    margin = top1 - top2
    rel_margin = margin / (abs(top1) + _EPS)
    mean = sum(scores) / len(scores)
    return {
        "top1": top1,
        "top2": top2,
        "margin": margin,
        "rel_margin": rel_margin,
        "mean": mean,
    }


def should_exit(scores: list[float], params: GateParams) -> bool:
    """Return True if this stage's results are confident enough to return.

    Exit iff: enough results, top-1 clears the absolute floor, AND the top-1 is
    relatively separated from the runner-up by at least ``rel_margin``. A lone
    result (``top2`` defaults to 0) has ``rel_margin == 1.0`` and so passes the
    separation check on floor alone.
    """
    if len(scores) < params.min_k:
        return False
    sig = confidence_signals(scores)
    if sig["top1"] < params.floor:
        return False
    return sig["rel_margin"] >= params.rel_margin
