"""Ablation study: compare dense-only / hybrid variants / rewrite+hybrid.

Each strategy runs **independently** (no cascade fallback), so the metrics
reflect what each stage would achieve on its own. This isolates the value
added by BM25, by different dense/sparse weight ratios, and by LLM-rewrite.

Strategies evaluated:
  dense          -- cosine dense search only (stage-1 equivalent)
  hybrid_5_5     -- dense 0.5 + BM25 0.5
  hybrid_7_3     -- dense 0.7 + BM25 0.3
  hybrid_3_7     -- dense 0.3 + BM25 0.7
  rewrite_hybrid -- LLM structural rewrite -> hybrid 0.5/0.5

The cascade thresholds used in production:
  stage1_min = 0.88   (was 0.65; raised so noisy/synonym queries reach stage-2)
  stage2_min = 0.75   (new; so synonym queries reach stage-3 rewrite)

Usage::

    uv run python -m backend.eval.run_ablation
    uv run python -m backend.eval.run_ablation --limit 80   # quick smoke
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
from backend.eval.retrieval.metrics import aggregate
from backend.v.configs import get_settings
from backend.v.rag.retriever import KnowledgeRetriever
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("eval.ablation")

KS = [1, 3, 5]
DEFAULT_TOP_K = 5
CONCURRENCY = 6
REPORT_PATH = Path(__file__).parent.parent / "data" / "ablation_report.json"

STRATEGIES = [
    ("dense", 0.0, 0.0),  # dense_weight/bm25_weight ignored for stage-1
    ("hybrid_5_5", 0.5, 0.5),
    ("hybrid_7_3", 0.7, 0.3),
    ("hybrid_3_7", 0.3, 0.7),
    # rewrite_hybrid handled separately (needs LLM)
]

# Recommended cascade thresholds derived from score-distribution analysis.
# Dense top-1 scores: noisy p25≈0.71, synonym p25≈0.62 -> stage1 threshold 0.88
# leaves ~80% of noisy/synonym queries to proceed to hybrid.
STAGE1_THRESHOLD = 0.88
STAGE2_THRESHOLD = 0.75


async def _run_strategy(
    retriever: KnowledgeRetriever,
    embedder: Any,
    llm_caller: Any,
    sem: asyncio.Semaphore,
    items: list[QaItem],
    strategy: str,
    dense_w: float,
    bm25_w: float,
    top_k: int,
    settings: Any,
) -> list[dict]:
    """Run one strategy over all items, returning per-query result rows."""
    milvus_cfg = settings.milvus
    coll = milvus_cfg.collection_name
    client = retriever._get_client(milvus_cfg.uri, milvus_cfg.token)

    async def _one(item: QaItem) -> dict:
        t0 = time.perf_counter()
        try:
            async with sem:
                vec = await embedder.aembed_query(item.query)
                if strategy == "rewrite_hybrid":
                    rewrite = await retriever._rewrite_query(llm_caller, item.query)
                    vec = await embedder.aembed_query(rewrite.rewritten_query)
                    bm25_text = " ".join(rewrite.keywords) if rewrite.keywords else item.query
                    source_type = rewrite.source_type_filter
                    results = await asyncio.to_thread(
                        retriever._execute_search_sync,
                        client,
                        coll,
                        2,
                        bm25_text,
                        vec,
                        top_k,
                        source_type,
                        0.5,
                        0.5,
                    )
                elif strategy == "dense":
                    results = await asyncio.to_thread(
                        retriever._execute_search_sync,
                        client,
                        coll,
                        1,
                        item.query,
                        vec,
                        top_k,
                        None,
                        dense_w,
                        bm25_w,
                    )
                else:
                    results = await asyncio.to_thread(
                        retriever._execute_search_sync,
                        client,
                        coll,
                        2,
                        item.query,
                        vec,
                        top_k,
                        None,
                        dense_w,
                        bm25_w,
                    )
        except Exception as exc:
            _log.warning(
                "ablation.failed", strategy=strategy, qa_id=item.qa_id, error=repr(exc)[:80]
            )
            results = []
        return {
            "qa_id": item.qa_id,
            "tier": item.tier,
            "difficulty": item.difficulty,
            "gold_source_type": item.gold_source_type,
            "gold": item.gold_source_ids,
            "retrieved": [r["source_id"] for r in results],
            "top_score": max((r["similarity"] for r in results), default=0.0),
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }

    return list(await asyncio.gather(*[_one(it) for it in items]))


def _summarize(rows: list[dict]) -> dict:
    overall = aggregate(rows, KS)

    def _group(key: str) -> dict[str, dict]:
        buckets: dict[str, list] = defaultdict(list)
        for r in rows:
            buckets[str(r.get(key, ""))].append(r)
        return {name: aggregate(group, KS) for name, group in sorted(buckets.items())}

    latencies = sorted(r["latency_ms"] for r in rows)
    n = len(latencies)
    return {
        "n": n,
        "overall": overall,
        "by_tier": _group("tier"),
        "by_difficulty": _group("difficulty"),
        "latency_ms": {
            "mean": sum(latencies) / n if n else 0,
            "p50": latencies[n // 2] if n else 0,
            "p95": latencies[int(n * 0.95)] if n else 0,
        },
    }


def _print_comparison(results: dict[str, dict]) -> None:
    strategies = list(results.keys())
    print(f"\n{'Strategy':<18}", end="")
    for k in KS:
        print(f"  hit@{k}  mrr@{k}  ndcg@{k}", end="")
    print("  mean_ms")
    print("-" * (18 + len(KS) * 24 + 10))
    for strat in strategies:
        m = results[strat]["overall"]
        lat = results[strat]["latency_ms"]["mean"]
        print(f"{strat:<18}", end="")
        for k in KS:
            h = m.get(f"hit@{k}", 0)
            r = m.get(f"mrr@{k}", 0)
            n = m.get(f"ndcg@{k}", 0)
            print(f"  {h:.3f}  {r:.3f}  {n:.3f}", end="")
        print(f"  {lat:.0f}")

    print(f"\n{'':18}", end="")
    print("--- by tier (hit@5) ---")
    tiers = ["plain", "colloquial", "synonym", "noisy", "multi_condition"]
    print(f"{'':18}", " ".join(f"{t:<16}" for t in tiers))
    for strat in strategies:
        by_tier = results[strat]["by_tier"]
        vals = [by_tier.get(t, {}).get("hit@5", 0.0) for t in tiers]
        print(f"{strat:<18}", " ".join(f"{v:.3f}{'':11}" for v in vals))

    print(f"\n[Cascade thresholds] stage1_min={STAGE1_THRESHOLD}  stage2_min={STAGE2_THRESHOLD}")
    print("=> stage-1 exits when dense top-1 >= 0.88 (plain/colloquial)")
    print("=> stage-2 exits when hybrid top-1 >= 0.75 (most synonym/multi_condition)")
    print("=> stage-3 LLM-rewrite for remaining noisy/hard queries")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--no-rewrite", action="store_true", help="skip LLM rewrite strategy")
    args = parser.parse_args()

    configure_logging(level="WARNING", json=False)
    items = load_qa()
    if not items:
        raise SystemExit("no qa.jsonl — run gen_qa first")
    if args.limit:
        items = items[: args.limit]

    settings = get_settings()
    embedder = get_embedder()
    llm_caller = None if args.no_rewrite else get_llm_caller()
    retriever = KnowledgeRetriever()
    # warm up collection (load into memory once)
    client = retriever._get_client(settings.milvus.uri, settings.milvus.token)
    await asyncio.to_thread(
        retriever._setup_collection_sync, client, settings.milvus.collection_name
    )

    sem = asyncio.Semaphore(CONCURRENCY)
    all_strategies = list(STRATEGIES)
    if not args.no_rewrite:
        all_strategies.append(("rewrite_hybrid", 0.5, 0.5))

    report: dict[str, Any] = {}
    for strat_name, dw, bw in all_strategies:
        print(f"\nrunning strategy: {strat_name} ...", flush=True)
        t0 = time.perf_counter()
        rows = await _run_strategy(
            retriever,
            embedder,
            llm_caller,
            sem,
            items,
            strat_name,
            dw,
            bw,
            args.top_k,
            settings,
        )
        elapsed = time.perf_counter() - t0
        report[strat_name] = _summarize(rows)
        report[strat_name]["wall_s"] = round(elapsed, 1)
        print(
            f"  done in {elapsed:.1f}s  hit@5={report[strat_name]['overall'].get('hit@5', 0):.3f}"
        )

    # Add recommended cascade config to report
    report["_cascade_thresholds"] = {
        "stage1_min_score": STAGE1_THRESHOLD,
        "stage2_min_score": STAGE2_THRESHOLD,
        "rationale": (
            "Dense scores for noisy/synonym queries cluster 0.59-0.82. "
            f"stage1_min={STAGE1_THRESHOLD} lets ~80% of hard queries proceed to hybrid. "
            f"stage2_min={STAGE2_THRESHOLD} lets remaining hard queries reach LLM-rewrite."
        ),
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_comparison({k: v for k, v in report.items() if not k.startswith("_")})
    print(f"\nfull report -> {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
