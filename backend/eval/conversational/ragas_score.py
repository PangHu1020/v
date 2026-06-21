"""RAGAS generation-quality scoring for the conversational eval.

The conversational harness measures *retrieval* (did the agent surface gold
products across the dialogue). This module adds *generation* quality via RAGAS
0.4's four metrics, scored per case after the dialogue runs:

- **Faithfulness** — is the agent's reply grounded in the retrieved contexts?
  (Catches the "no search → hallucinated products" failure directly.)
- **AnswerRelevancy** — does the reply actually address the customer's need?
- **ContextRecall** — do the retrieved contexts cover the reference facts?
- **AnswerCorrectness** — does the reply match the reference answer?

``ConvCase`` carries no hand-written reference answer, so we synthesise one
mechanically from the gold products (the same code-enumerated gold the any-of
retrieval scoring uses) — never LLM-judged, so it can't silently drift.

Multi-turn → single RAGAS sample mapping (per case):
- ``user_input``  = the case ``hidden_need`` (the canonical, de-noised intent)
- ``response``    = the final agent reply
- ``retrieved_contexts`` = union of chunk texts surfaced across all search calls
- ``reference``   = synthesised gold-product summary

The judge LLM is an OpenAI-compatible endpoint independent of the agent under
test (see run_conv_eval ``--judge-*`` / ``EVAL_JUDGE_*`` env), so a slow or weak
agent model never contaminates the scores.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from backend.eval.common import Product
from backend.v.utils.logging import get_logger

_log = get_logger("eval.conv.ragas")

# Lines the search tool emits look like ``- [product:P001] <chunk text>``.
_CTX_LINE = re.compile(r"^-\s*\[(?:product|faq):[^\]]+\]\s*(.+)$")


def extract_contexts(tool_outputs: list[str]) -> list[str]:
    """Pull the per-item chunk texts out of the search tool's rendered output.

    The ``search`` tool formats hits as ``- [product:P001] <text>`` lines under
    a header. We keep the ``<text>`` payloads (de-duplicated, first-seen order)
    as the ``retrieved_contexts`` RAGAS scores Faithfulness / ContextRecall on.
    """
    seen: dict[str, None] = {}
    for out in tool_outputs:
        if not out:
            continue
        for line in out.splitlines():
            m = _CTX_LINE.match(line.strip())
            if m:
                text = m.group(1).strip()
                if text and text not in seen:
                    seen[text] = None
    return list(seen)


def build_reference(gold_ids: list[str], products: dict[str, Product]) -> str:
    """Synthesise a natural-language reference answer from the gold products.

    Lists each gold product's name + price + one-line description, so:
    - ContextRecall can check the retrieved contexts cover these facts, and
    - AnswerCorrectness can check the agent's reply names the right products.

    Mechanically derived from the code-enumerated gold set — reproducible, no
    LLM judgement, cannot under-label.
    """
    lines: list[str] = []
    for gid in gold_ids:
        p = products.get(gid)
        if not p:
            continue
        desc = p.description.strip()
        lines.append(f"{p.product_name}（{p.price:g}元，{p.brand}）：{desc}")
    if not lines:
        return ""
    return "符合客户需求的商品包括：\n" + "\n".join(f"- {ln}" for ln in lines)


def _kwargs_for(metric_name: str, sample: dict[str, Any]) -> dict[str, Any]:
    """Map a RAGAS collections metric to its required ``ascore`` kwargs."""
    if metric_name == "Faithfulness":
        return {
            "user_input": sample["user_input"],
            "response": sample["response"],
            "retrieved_contexts": sample["retrieved_contexts"],
        }
    if metric_name == "ContextRecall":
        return {
            "user_input": sample["user_input"],
            "retrieved_contexts": sample["retrieved_contexts"],
            "reference": sample["reference"],
        }
    if metric_name == "AnswerRelevancy":
        return {"user_input": sample["user_input"], "response": sample["response"]}
    # AnswerCorrectness
    return {
        "user_input": sample["user_input"],
        "response": sample["response"],
        "reference": sample["reference"],
    }


def scorable(sample: dict[str, Any]) -> bool:
    """A sample is RAGAS-scorable only with a non-empty response AND contexts.

    Cases where the agent never searched (no contexts) or never replied are
    excluded — scoring them would inject spurious zeros that conflate an
    agent-behaviour gap (didn't call search) with a generation-quality gap.
    """
    return bool(sample.get("response")) and bool(sample.get("retrieved_contexts"))


async def score_cases(
    samples: list[dict[str, Any]],
    *,
    judge_base_url: str,
    judge_api_key: str,
    judge_model: str,
    embed_base_url: str,
    embed_api_key: str,
    embed_model: str,
    concurrency: int = 4,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Score samples with all four RAGAS metrics.

    Args:
        samples: dicts with keys ``case_id``, ``user_input``, ``response``,
            ``retrieved_contexts``, ``reference``.
        judge_*: OpenAI-compatible endpoint for the judge LLM.
        embed_*: OpenAI-compatible endpoint for the embedder (relevancy/correctness).
        concurrency: max concurrent judge calls.

    Returns:
        ``(aggregate, per_case)`` where ``aggregate`` maps metric name → mean
        score over scorable samples, and ``per_case`` is a list of
        ``{case_id, <metric>: score}`` dicts.
    """
    from backend.eval.generate._ragas_setup import build_ragas_metrics

    metrics = build_ragas_metrics(
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        judge_model=judge_model,
        embed_base_url=embed_base_url,
        embed_api_key=embed_api_key,
        embed_model=embed_model,
    )
    metric_names = [type(m).__name__ for m in metrics]
    eligible = [s for s in samples if scorable(s)]
    _log.info("eval.conv.ragas.start", eligible=len(eligible), total=len(samples))
    if not eligible:
        return {n: 0.0 for n in metric_names}, []

    sem = asyncio.Semaphore(concurrency)

    async def _one(metric: Any, sample: dict[str, Any]) -> float | None:
        async with sem:
            try:
                res = await metric.ascore(**_kwargs_for(type(metric).__name__, sample))
                return float(res.value) if res is not None else None
            except Exception as exc:
                _log.warning(
                    "eval.conv.ragas.cell_failed",
                    metric=type(metric).__name__,
                    case_id=sample.get("case_id"),
                    error=type(exc).__name__,
                )
                return None

    flat = await asyncio.gather(*[_one(m, s) for m in metrics for s in eligible])

    # Reshape flat [m0s0, m0s1, ..., m1s0, ...] → per-metric columns.
    n = len(eligible)
    aggregate: dict[str, float] = {}
    per_case: list[dict[str, Any]] = [{"case_id": s["case_id"]} for s in eligible]
    for mi, name in enumerate(metric_names):
        col = flat[mi * n : (mi + 1) * n]
        vals = [v for v in col if v is not None]
        aggregate[name] = round(sum(vals) / len(vals), 4) if vals else 0.0
        for si, v in enumerate(col):
            if v is not None:
                per_case[si][name] = round(v, 4)
    return aggregate, per_case
