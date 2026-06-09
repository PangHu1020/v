"""Calibrate the cascade confidence gate from the labelled QA set.

The gate (``backend.v.rag.gate``) decides when a cascade stage may exit instead
of escalating. Its params — per-stage ``floor`` + ``rel_margin`` — should be
derived from data, not hand-picked, so they survive an embedding-model or
corpus change.

Method
------
For each stage (1 = dense, 2 = hybrid) we run every QA query, collect the
result score list, and label the query ``gold_present`` = the gold doc is in
the returned top-k. We then grid-search ``(floor, rel_margin)`` to maximise
**Youden's J** = TPR − FPR of the decision "exit here" against the label
"gold_present":

- A query where gold IS present and we exit  → true positive  (good, saved an escalation).
- A query where gold is ABSENT and we exit   → false positive (bad, returned a miss).
- A query where gold is ABSENT and we escalate → true negative (good, recovered downstream).

Maximising J picks the operating point that best separates "trust this stage"
from "escalate", balancing wasted escalations against returned misses.

Cost: embeds each query once (no chat-LLM calls — stages 1/2 are pure
retrieval). Run after seeding Milvus.

Usage::

    uv run python -m backend.eval.retrieval.calibrate            # report only
    uv run python -m backend.eval.retrieval.calibrate --write    # write config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from backend.eval.common import get_embedder, load_qa
from backend.v.configs import get_settings
from backend.v.rag.gate import GateParams, should_exit
from backend.v.rag.retriever import KnowledgeRetriever
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("eval.calibrate")

CONCURRENCY = 4
TOP_K = 5
_FLOOR_GRID = [round(x / 100, 2) for x in range(0, 96, 5)]  # 0.00 .. 0.95
_MARGIN_GRID = [round(x / 100, 2) for x in range(0, 51, 2)]  # 0.00 .. 0.50


async def _collect_all_scores(
    retriever: KnowledgeRetriever,
    embedder: Any,
    settings: Any,
) -> dict[int, list[tuple[list[float], bool]]]:
    """Embed each query ONCE, run both stages, return {stage: [(scores, gold)]}.

    Embedding is the only paid call here (no chat-LLM); embedding once and
    reusing the vector for stage-1 and stage-2 halves the cost.
    """
    items = load_qa()
    milvus = settings.milvus
    client = retriever._get_client(milvus.uri, milvus.token)
    await asyncio.to_thread(retriever._setup_collection_sync, client, milvus.collection_name)
    rag = settings.rag
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(item) -> dict[int, tuple[list[float], bool]]:
        async with sem:
            vec = await embedder.aembed_query(item.query)
        gold = set(item.gold_source_ids)
        out: dict[int, tuple[list[float], bool]] = {}
        for stage in (1, 2):
            rows = await asyncio.to_thread(
                retriever._execute_search_sync,
                client,
                milvus.collection_name,
                stage,
                item.query,
                vec,
                TOP_K,
                None,
                rag.dense_weight,
                rag.bm25_weight,
            )
            scores = [r["similarity"] for r in rows]
            present = any(r["source_id"] in gold for r in rows)
            out[stage] = (scores, present)
        return out

    per_query = await asyncio.gather(*[one(it) for it in items])
    return {
        1: [q[1] for q in per_query],
        2: [q[2] for q in per_query],
    }


def _youden_grid(data: list[tuple[list[float], bool]], min_k: int) -> dict[str, Any]:
    """Grid-search (floor, rel_margin) maximising Youden's J = TPR - FPR."""
    pos = sum(1 for _, g in data if g)
    neg = len(data) - pos
    best = {"floor": 0.0, "rel_margin": 0.0, "j": -1.0, "tpr": 0.0, "fpr": 0.0}
    for floor in _FLOOR_GRID:
        for margin in _MARGIN_GRID:
            params = GateParams(floor=floor, rel_margin=margin, min_k=min_k)
            tp = fp = 0
            for scores, gold in data:
                if should_exit(scores, params):
                    if gold:
                        tp += 1
                    else:
                        fp += 1
            tpr = tp / pos if pos else 0.0
            fpr = fp / neg if neg else 0.0
            j = tpr - fpr
            if j > best["j"]:
                best = {
                    "floor": floor,
                    "rel_margin": margin,
                    "j": round(j, 4),
                    "tpr": round(tpr, 4),
                    "fpr": round(fpr, 4),
                }
    best["n"] = len(data)
    best["gold_present"] = pos
    return best


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write tuned params into config.yaml")
    args = parser.parse_args()

    configure_logging(level="WARNING", json=False)
    settings = get_settings()
    embedder = get_embedder()
    retriever = KnowledgeRetriever()

    print("collecting stage-1 + stage-2 score distributions (embed once per query) …")
    scores = await _collect_all_scores(retriever, embedder, settings)
    s1, s2 = scores[1], scores[2]

    min_k = settings.rag.min_k
    best1 = _youden_grid(s1, min_k)
    best2 = _youden_grid(s2, min_k)

    print("\n=== calibration result (Youden's J) ===")
    print(
        f"stage1: floor={best1['floor']} rel_margin={best1['rel_margin']} "
        f"J={best1['j']} (TPR={best1['tpr']} FPR={best1['fpr']}, "
        f"gold_present={best1['gold_present']}/{best1['n']})"
    )
    print(
        f"stage2: floor={best2['floor']} rel_margin={best2['rel_margin']} "
        f"J={best2['j']} (TPR={best2['tpr']} FPR={best2['fpr']}, "
        f"gold_present={best2['gold_present']}/{best2['n']})"
    )

    yaml_block = (
        "rag:\n"
        f"  stage1_floor: {best1['floor']}\n"
        f"  stage1_rel_margin: {best1['rel_margin']}\n"
        f"  stage2_floor: {best2['floor']}\n"
        f"  stage2_rel_margin: {best2['rel_margin']}\n"
    )
    print("\nsuggested config.yaml block:\n" + yaml_block)

    if args.write:
        import yaml

        from backend.v.configs.base import _config_path

        path = _config_path()

        def _read_write() -> None:
            try:
                with open(path, encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except FileNotFoundError:
                data = {}
            rag = dict(data.get("rag") or {})
            rag.update(
                stage1_floor=best1["floor"],
                stage1_rel_margin=best1["rel_margin"],
                stage2_floor=best2["floor"],
                stage2_rel_margin=best2["rel_margin"],
            )
            data["rag"] = rag
            with open(path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)

        await asyncio.to_thread(_read_write)
        print(f"\nwrote tuned rag params -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
