"""Generate-quality eval using RAGAS.

Metrics: Faithfulness / ContextRecall / AnswerRelevancy / AnswerCorrectness.
Calls the real agent for each QA item, collects retrieved contexts + response,
then scores with RAGAS.

Usage::

    uv run python -m backend.eval.generate.run_generate_eval
    uv run python -m backend.eval.generate.run_generate_eval --limit 50
    uv run python -m backend.eval.generate.run_generate_eval --difficulty hard
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from backend.eval.common import load_qa
from backend.eval.generate._ragas_setup import build_ragas_metrics
from backend.v.configs import get_settings
from backend.v.models.factory import get_chat_model, get_embedding
from backend.v.rag.retriever import KnowledgeRetriever
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("eval.generate")
REPORT_PATH = Path(__file__).parent.parent / "data" / "generate_eval_report.json"
CONCURRENCY = 4


async def _retrieve_and_generate(
    retriever: KnowledgeRetriever,
    embedder: Any,
    llm: Any,
    sem: asyncio.Semaphore,
    item,
    settings: Any,
) -> dict[str, Any] | None:
    """Retrieve context, generate answer, return ragas-ready row."""
    try:
        async with sem:
            results = await retriever.retrieve(
                embedder=embedder,
                query=item.query,
                top_k=5,
                settings=settings,
            )
            contexts = [r["text"] for r in results]

            # Single-turn: system + user message, no tools, just direct answer
            from langchain_core.messages import SystemMessage

            ctx_text = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
            prompt = [
                SystemMessage(
                    content="你是一名电商客服助理。根据以下检索到的参考资料回答客户问题。"
                ),
                HumanMessage(content=f"参考资料：\n{ctx_text}\n\n客户问题：{item.query}"),
            ]
            resp = await llm.ainvoke(prompt)
            answer = resp.content if isinstance(resp.content, str) else ""

        return {
            "qa_id": item.qa_id,
            "difficulty": item.difficulty,
            "user_input": item.query,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": item.answer,
        }
    except Exception as exc:
        _log.warning("generate.row_failed", qa_id=item.qa_id, error=type(exc).__name__)
        return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument(
        "--model",
        default="main_fallback",
        choices=["main_primary", "main_fallback"],
        help="LLM role slot for generation.",
    )
    parser.add_argument(
        "--gen-model",
        default=None,
        metavar="MODEL_NAME",
        help="Override model name (e.g. qwen3-14b). Same API URL/key from .env."
        " Without this flag the name from --model slot is used.",
    )
    args = parser.parse_args()

    configure_logging(level="WARNING", json=False)
    settings = get_settings()
    items = load_qa()

    if args.difficulty != "all":
        items = [it for it in items if it.difficulty == args.difficulty]
    if args.limit:
        items = items[: args.limit]

    if not items:
        raise SystemExit("no qa items — check filters")

    embedder = get_embedding(settings.llm, settings.embedding)

    # Build the generation LLM, optionally overriding the model name.
    llm_settings = settings.llm
    if args.gen_model:
        import copy

        llm_settings = copy.copy(settings.llm)
        llm_settings.main_primary = args.gen_model
        llm_settings.main_fallback = args.gen_model
    llm = get_chat_model(llm_settings, args.model)
    gen_model_name = args.gen_model or getattr(
        settings.llm, args.model.replace("-", "_"), args.model
    )

    retriever = KnowledgeRetriever()
    sem = asyncio.Semaphore(CONCURRENCY)

    print(f"generating answers for {len(items)} queries …")
    t0 = time.perf_counter()
    rows = await asyncio.gather(
        *[_retrieve_and_generate(retriever, embedder, llm, sem, it, settings) for it in items]
    )
    rows = [r for r in rows if r is not None]
    gen_s = time.perf_counter() - t0
    print(f"  done in {gen_s:.1f}s, {len(rows)} valid rows")

    # Build RAGAS dataset
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            retrieved_contexts=r["retrieved_contexts"],
            response=r["response"],
            reference=r["reference"],
        )
        for r in rows
    ]
    dataset = EvaluationDataset(samples=samples)

    metrics = build_ragas_metrics(llm, embedder)

    print("running RAGAS evaluation …")
    from ragas import aevaluate

    t1 = time.perf_counter()
    result = await aevaluate(dataset, metrics=metrics, raise_exceptions=False)
    ragas_s = time.perf_counter() - t1

    scores = result.to_pandas().mean(numeric_only=True).to_dict()
    print(f"  done in {ragas_s:.1f}s")

    report = {
        "n": len(rows),
        "gen_model": gen_model_name,
        "difficulty_filter": args.difficulty,
        "generation_wall_s": round(gen_s, 1),
        "ragas_wall_s": round(ragas_s, 1),
        "scores": {k: round(float(v), 4) for k, v in scores.items() if k != "qa_id"},
    }

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== RAGAS generate eval — {len(rows)} items ===")
    for k, v in report["scores"].items():
        print(f"  {k:<25} {v:.4f}")
    print(f"\nreport -> {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
