"""End-to-end retrieval evaluation harness.

Runs the **real** ``KnowledgeRetriever.retrieve()`` (the same code path the
production ``search`` tool uses) over every item in ``data/qa.jsonl``, against
the Milvus collection seeded by ``seed_milvus.py``. For each query it records
the ranked ``source_id`` list and which cascade stage produced it (via the
retriever's ``trace`` hook), then reports:

- Overall recall@k / MRR@k / nDCG@k / hit@k.
- The same metrics broken down by difficulty tier.
- The same metrics broken down by gold ``source_type`` (product vs faq).
- Cascade-stage distribution (how often stage-1 dense suffices vs needs
  hybrid vs needs LLM-rewrite) and per-stage hit@k.
- Mean retrieval latency.

Queries run with bounded concurrency. A JSON report is written to
``data/eval_report.json`` and a human summary is printed.

Usage::

    uv run python -m backend.eval.run_eval
    uv run python -m backend.eval.run_eval --limit 50      # quick smoke
    uv run python -m backend.eval.run_eval --top-k 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from backend.eval.common import QaItem, get_embedder, get_llm_caller, load_qa
from backend.eval.metrics import aggregate
from backend.v.configs import get_settings
from backend.v.rag.retriever import KnowledgeRetriever
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("eval.run_eval")

KS = [1, 3, 5]
DEFAULT_TOP_K = 5
CONCURRENCY = 8
REPORT_PATH = Path(__file__).parent / "data" / "eval_report.json"


async def _eval_one(
    retriever: KnowledgeRetriever,
    embedder: Any,
    llm_caller: Any,
    sem: asyncio.Semaphore,
    item: QaItem,
    top_k: int,
    settings: Any,
) -> dict[str, Any]:
    trace: dict[str, Any] = {}
    t0 = time.perf_counter()
    async with sem:
        try:
            results = await retriever.retrieve(
                embedder=embedder,
                llm_caller=llm_caller,
                query=item.query,
                top_k=top_k,
                source_type=None,  # eval the router end-to-end, no filter hint
                settings=settings,
                trace=trace,
            )
        except Exception as exc:
            _log.warning("eval.query_failed", qa_id=item.qa_id, error=type(exc).__name__)
            results = []
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "qa_id": item.qa_id,
        "tier": item.tier,
        "gold_source_type": item.gold_source_type,
        "gold": item.gold_source_ids,
        "retrieved": [r["source_id"] for r in results],
        "stage": trace.get("stage", 0),
        "latency_ms": latency_ms,
    }


def _report(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    overall = aggregate(rows, KS)

    def _group(key: str) -> dict[str, dict[str, float]]:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            buckets[str(r[key])].append(r)
        return {name: aggregate(group, KS) for name, group in sorted(buckets.items())}

    stage_counts: dict[int, int] = defaultdict(int)
    for r in rows:
        stage_counts[r["stage"]] += 1
    stage_dist = {f"stage_{s}": stage_counts[s] for s in sorted(stage_counts)}

    latencies = sorted(r["latency_ms"] for r in rows)
    n = len(latencies)
    return {
        "n_queries": n,
        "top_k": top_k,
        "overall": overall,
        "by_tier": _group("tier"),
        "by_source_type": _group("gold_source_type"),
        "by_stage": {
            "distribution": stage_dist,
            "metrics": _group("stage"),
        },
        "latency_ms": {
            "mean": sum(latencies) / n if n else 0.0,
            "p50": latencies[n // 2] if n else 0.0,
            "p95": latencies[int(n * 0.95)] if n else 0.0,
        },
    }


def _print_summary(report: dict[str, Any]) -> None:
    def _fmt(m: dict[str, float]) -> str:
        return " ".join(f"{k}={v:.3f}" for k, v in m.items())

    print(f"\n=== RAG retrieval eval — {report['n_queries']} queries, top_k={report['top_k']} ===")
    print("OVERALL :", _fmt(report["overall"]))
    print("\nBY TIER:")
    for tier, m in report["by_tier"].items():
        print(f"  {tier:<16}", _fmt(m))
    print("\nBY SOURCE_TYPE:")
    for st, m in report["by_source_type"].items():
        print(f"  {st:<16}", _fmt(m))
    print("\nCASCADE STAGE DISTRIBUTION:", report["by_stage"]["distribution"])
    lat = report["latency_ms"]
    print(f"\nLATENCY ms: mean={lat['mean']:.0f} p50={lat['p50']:.0f} p95={lat['p95']:.0f}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="evaluate only first N items")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    configure_logging(level="WARNING", json=False)
    items = load_qa()
    if not items:
        raise SystemExit("no qa.jsonl — run gen_qa first")
    if args.limit:
        items = items[: args.limit]

    settings = get_settings()
    embedder = get_embedder()
    llm_caller = get_llm_caller()
    retriever = KnowledgeRetriever()
    sem = asyncio.Semaphore(CONCURRENCY)

    t0 = time.perf_counter()
    rows = await asyncio.gather(
        *[_eval_one(retriever, embedder, llm_caller, sem, it, args.top_k, settings) for it in items]
    )
    elapsed = time.perf_counter() - t0

    report = _report(list(rows), args.top_k)
    report["wall_seconds"] = round(elapsed, 1)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(report)
    print(f"\nfull report -> {REPORT_PATH}  (wall {elapsed:.0f}s)")


if __name__ == "__main__":
    asyncio.run(main())
