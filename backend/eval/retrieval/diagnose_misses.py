"""Isolate WHY the 7 retrieval-miss cases missed: query quality vs recall vs rerank.

For each miss case, run the retriever with the case's hidden_need (the clean,
ground-truth intent) and report where each gold id lands:
  - in the hybrid candidate pool (pre-rerank rank)
  - after rerank (final rank)
This separates: bad agent query / recall gap / rerank demotion / over-broad gold.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from backend.eval.common import get_embedder
from backend.v.configs import get_settings
from backend.v.rag.rerank import build_reranker
from backend.v.rag.retriever import KnowledgeRetriever

MISS_IDS = ["CONV0001", "CONV0002", "CONV0010", "CONV0011", "CONV0021", "CONV0023", "CONV0030"]


def _load_cases() -> dict:
    cases = {}
    with open("backend/eval/data/conv_cases.jsonl") as f:
        for line in f:
            c = json.loads(line)
            cases[c["case_id"]] = c
    return cases


def _load_products() -> dict:
    prods = {}
    with open("backend/eval/data/products.jsonl") as f:
        for line in f:
            p = json.loads(line)
            prods[p["product_id"]] = p
    return prods


def _rank_of(gold: set[str], rows: list[dict]) -> dict[str, int]:
    """Return {gold_id: 1-based rank or -1 if absent} for each gold id."""
    pos = {r["source_id"]: i + 1 for i, r in enumerate(rows)}
    return {g: pos.get(g, -1) for g in gold}


async def main() -> None:
    cases = _load_cases()
    settings = get_settings()
    embedder = get_embedder()
    retriever = KnowledgeRetriever()

    # Prime the Milvus client once.
    client = retriever._get_client(settings.milvus.uri, settings.milvus.token)
    coll = settings.milvus.collection_name
    import asyncio as _aio

    await _aio.to_thread(retriever._setup_collection_sync, client, coll)

    async def _search_with_filter(query: str, filter_expr: str | None, k: int) -> list[dict]:
        """Hybrid search with an explicit Milvus filter expr, then rerank to k."""
        from pymilvus import AnnSearchRequest, WeightedRanker

        qv = await embedder.aembed_query(query)
        rag = settings.rag
        cand_k = min(max(k, rag.rerank_candidates), 80)

        def _run():
            req_dense = AnnSearchRequest(
                data=[qv],
                anns_field="embedding",
                param={"metric_type": "COSINE"},
                limit=cand_k,
                expr=filter_expr,
            )
            req_sparse = AnnSearchRequest(
                data=[query],
                anns_field="sparse_embedding",
                param={"metric_type": "BM25"},
                limit=cand_k,
                expr=filter_expr,
            )
            res = client.hybrid_search(
                collection_name=coll,
                reqs=[req_dense, req_sparse],
                ranker=WeightedRanker(rag.dense_weight, rag.bm25_weight),
                limit=cand_k,
                output_fields=["source_type", "source_id", "text", "metadata"],
            )
            return res[0] if res else []

        hits = await _aio.to_thread(_run)

        def _entity(h: Any) -> dict:
            return h.get("entity", {}) if isinstance(h, dict) else getattr(h, "entity", {})

        rows = [
            {"source_id": _entity(h).get("source_id"), "text": _entity(h).get("text", "")}
            for h in hits
        ]
        # Rerank the filtered pool.
        if rag.rerank_enabled and rows:
            reranker = build_reranker(
                url=rag.rerank_url, model=rag.rerank_model, api_key=rag.rerank_api_key
            )
            rows = await reranker.rerank(query, rows, top_n=k)
        return rows[:k]

    for cid in MISS_IDS:
        c = cases[cid]
        gold = set(c["gold_source_ids"])
        need = c["hidden_need"]
        constraint = c["constraint"]
        print(f"\n{'=' * 70}\n[{cid}] {c['persona']} / {constraint.get('category')}")
        print(f"  constraint: {constraint}")

        # Build the metadata filter from the case constraint (what a structured
        # LLM output would emit: category + price band).
        cat = constraint.get("category")
        pmin = constraint.get("price_min")
        pmax = constraint.get("price_max")
        clauses = []
        if cat:
            clauses.append(f'metadata["category"] == "{cat}"')
        if pmin is not None:
            clauses.append(f'metadata["price"] >= {pmin}')
        if pmax is not None:
            clauses.append(f'metadata["price"] <= {pmax}')
        meta_expr = " and ".join(clauses) if clauses else None

        # Baseline: no filter, rerank to top5.
        base = await _search_with_filter(need, None, 5)
        base_ranks = _rank_of(gold, base)
        # With metadata filter, rerank to top5.
        filt = await _search_with_filter(need, meta_expr, 5)
        filt_ranks = _rank_of(gold, filt)

        base_hit = sum(1 for v in base_ranks.values() if v > 0)
        filt_hit = sum(1 for v in filt_ranks.values() if v > 0)
        print(f"  filter: {meta_expr}")
        print(f"  无过滤   top5命中gold: {base_hit}/{len(gold)}  {base_ranks}")
        print(f"  meta过滤 top5命中gold: {filt_hit}/{len(gold)}  {filt_ranks}")
        verdict = (
            "✅ 改善" if filt_hit > base_hit else ("持平" if filt_hit == base_hit else "❌ 变差")
        )
        print(f"  → {verdict} ({base_hit}→{filt_hit})")


if __name__ == "__main__":
    asyncio.run(main())
