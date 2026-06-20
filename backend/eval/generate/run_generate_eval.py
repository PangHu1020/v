"""Generate-quality eval using RAGAS.

Metrics: Faithfulness / ContextRecall / AnswerRelevancy / AnswerCorrectness.
Calls the real retriever + an LLM for each QA item, then scores with RAGAS.

Usage::

    # default model (main_fallback from .env)
    uv run python -m backend.eval.generate.run_generate_eval --limit 50

    # Qwen3 models on DashScope (thinking disabled)
    uv run python -m backend.eval.generate.run_generate_eval --gen-model qwen3-8b
    uv run python -m backend.eval.generate.run_generate_eval --gen-model qwen3-14b

    Reports land in  eval/outputs/generate_<model>_<difficulty>.json
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.eval.common import load_qa
from backend.eval.generate._ragas_setup import build_ragas_metrics
from backend.v.configs import get_settings
from backend.v.models.factory import get_chat_model, get_embedding
from backend.v.rag.retriever import KnowledgeRetriever
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("eval.generate")
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
CONCURRENCY = 4

# Qwen3 models output <think>…</think> unless disabled.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


async def _retrieve_and_generate(
    retriever: KnowledgeRetriever,
    embedder: Any,
    llm: Any,
    sem: asyncio.Semaphore,
    item: Any,
    settings: Any,
) -> dict[str, Any] | None:
    try:
        async with sem:
            results = await retriever.retrieve(
                embedder=embedder, query=item.query, top_k=5, settings=settings
            )
            contexts = [r["text"] for r in results]
            ctx_text = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
            prompt = [
                SystemMessage(
                    content="你是一名电商客服助理。根据以下检索到的参考资料回答客户问题。"
                ),
                HumanMessage(content=f"参考资料：\n{ctx_text}\n\n客户问题：{item.query}"),
            ]
            resp = await llm.ainvoke(prompt)
            raw = resp.content if isinstance(resp.content, str) else ""
            answer = _strip_thinking(raw)

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
        help="LLM role slot (selects base_url + api_key from .env).",
    )
    parser.add_argument(
        "--gen-model",
        default=None,
        metavar="MODEL_NAME",
        help="Override model name (e.g. qwen3-8b, qwen3-14b). API URL/key from .env.",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable chain-of-thought for Qwen3 models (passes enable_thinking=False).",
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

    llm_settings = settings.llm
    if args.gen_model:
        llm_settings = copy.copy(settings.llm)
        llm_settings.model = args.gen_model
        llm_settings.model_fallback = args.gen_model

    llm = get_chat_model(llm_settings, args.model)
    if args.no_think:
        # DashScope-compatible way to disable Qwen3 thinking mode
        llm = llm.bind(extra_body={"enable_thinking": False})

    gen_model_name = args.gen_model or (
        settings.llm.model if args.model == "main_primary" else settings.llm.model_fallback
    )

    retriever = KnowledgeRetriever()
    sem = asyncio.Semaphore(CONCURRENCY)

    print(f"model={gen_model_name}  no_think={args.no_think}  n={len(items)}")
    t0 = time.perf_counter()
    rows = await asyncio.gather(
        *[_retrieve_and_generate(retriever, embedder, llm, sem, it, settings) for it in items]
    )
    rows = [r for r in rows if r is not None]
    gen_s = time.perf_counter() - t0
    print(f"  generated {len(rows)} answers in {gen_s:.1f}s")

    metrics = build_ragas_metrics(
        judge_base_url=settings.llm.base_url_fallback,
        judge_api_key=settings.llm.api_key_fallback,
        judge_model=settings.llm.model_fallback,  # fallback endpoint judges
        embed_base_url=settings.embedding.base_url,
        embed_api_key=settings.embedding.api_key,
        embed_model=settings.embedding.model,
    )

    print("running RAGAS …")
    t1 = time.perf_counter()

    # collections metrics use ascore() with different kwargs per metric type
    def _kwargs(m, r):
        name = type(m).__name__
        if name == "Faithfulness":
            return {
                "user_input": r["user_input"],
                "response": r["response"],
                "retrieved_contexts": r["retrieved_contexts"],
            }
        if name == "ContextRecall":
            return {
                "user_input": r["user_input"],
                "retrieved_contexts": r["retrieved_contexts"],
                "reference": r["reference"],
            }
        if name == "AnswerRelevancy":
            return {"user_input": r["user_input"], "response": r["response"]}
        # AnswerCorrectness
        return {
            "user_input": r["user_input"],
            "response": r["response"],
            "reference": r["reference"],
        }

    metric_names = [type(m).__name__ for m in metrics]
    ragas_sem = asyncio.Semaphore(CONCURRENCY)

    async def score_one(m, r):
        async with ragas_sem:
            return await m.ascore(**_kwargs(m, r))

    all_results = await asyncio.gather(
        *[score_one(m, r) for m in metrics for r in rows],
        return_exceptions=True,
    )
    ragas_s = time.perf_counter() - t1

    # reshape: all_results = [m0r0, m0r1, ..., m1r0, ...]
    n = len(rows)
    scores: dict[str, float] = {}
    for mi, name in enumerate(metric_names):
        vals = [
            v.value
            for v in all_results[mi * n : (mi + 1) * n]
            if not isinstance(v, Exception) and v is not None
        ]
        scores[name] = round(sum(vals) / len(vals), 4) if vals else 0.0

    report = {
        "gen_model": gen_model_name,
        "no_think": args.no_think,
        "difficulty_filter": args.difficulty,
        "n": len(rows),
        "generation_wall_s": round(gen_s, 1),
        "ragas_wall_s": round(ragas_s, 1),
        "scores": scores,
    }

    safe_name = re.sub(r"[^a-z0-9_-]", "_", gen_model_name.lower())
    out_path = OUTPUTS_DIR / f"generate_{safe_name}_{args.difficulty}.json"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== RAGAS generate eval — {gen_model_name} ({args.difficulty}) ===")
    for k, v in report["scores"].items():
        print(f"  {k:<25} {v:.4f}")
    print(f"\nreport -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
