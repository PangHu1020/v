"""System performance eval: E2E latency + token cost estimation.

Calls the real graph (enter → compress → intent → agent → reflect → exit)
with real LLM + Milvus retrieval for each QA query. Captures:

- E2E wall time per turn
- Per-stage breakdown (intent, agent, reflection, tools)
- Input/output tokens per turn (from LLMResult.latency_ms + LangSmith callback)
- Estimated cost (configured price per 1M tokens)

Usage::

    uv run python -m backend.eval.system.run_system_eval
    uv run python -m backend.eval.system.run_system_eval --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from backend.eval.common import TokenCounter, load_qa
from backend.v.agents.graph import build_graph
from backend.v.configs import get_settings
from backend.v.models.factory import get_embedding
from backend.v.models.llm_caller import LLMCaller
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("eval.system")
REPORT_PATH = Path(__file__).parent.parent / "data" / "system_eval_report.json"

# ── Price table (RMB per 1M tokens) — update as needed ──────────────────────
PRICE_PER_1M = {
    "input": 0.5,  # deepseek-v4-flash input
    "output": 1.5,  # deepseek-v4-flash output
}

CONCURRENCY = 4


async def _run_one(
    graph: Any,
    embedder: Any,
    llm_caller: LLMCaller,
    settings: Any,
    session_id: str,
    item,
) -> dict[str, Any]:
    counter = TokenCounter()
    config = {
        "configurable": {
            "thread_id": session_id,
            "llm_caller": llm_caller,
            "embedder": embedder,
            "settings": settings,
            "compression_threshold_tokens": 0,  # disable compression for eval
        },
        "callbacks": [counter],
    }
    input_state = {"messages": [HumanMessage(content=item.query)]}
    t0 = time.perf_counter()
    try:
        await graph.ainvoke(input_state, config=config)
    except Exception as exc:
        _log.warning("system.turn_failed", qa_id=item.qa_id, error=type(exc).__name__)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "qa_id": item.qa_id,
        "difficulty": item.difficulty,
        "elapsed_ms": round(elapsed_ms, 1),
        "prompt_tokens": counter.prompt_tokens,
        "completion_tokens": counter.completion_tokens,
        "total_tokens": counter.prompt_tokens + counter.completion_tokens,
    }


def _cost_rmb(prompt: int, completion: int) -> float:
    return (prompt * PRICE_PER_1M["input"] + completion * PRICE_PER_1M["output"]) / 1_000_000


def _summarize(rows: list[dict]) -> dict:
    latencies = sorted(r["elapsed_ms"] for r in rows)
    n = len(latencies)
    total_in = sum(r["prompt_tokens"] for r in rows)
    total_out = sum(r["completion_tokens"] for r in rows)
    cost = _cost_rmb(total_in, total_out)

    def _by(key: str) -> dict:
        buckets: dict[str, list] = {}
        for r in rows:
            buckets.setdefault(str(r.get(key, "")), []).append(r["elapsed_ms"])
        return {
            k: {
                "mean_ms": round(statistics.mean(v), 0),
                "p95_ms": round(sorted(v)[int(len(v) * 0.95)], 0),
            }
            for k, v in sorted(buckets.items())
        }

    return {
        "n": n,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 0),
            "median": round(statistics.median(latencies), 0),
            "p95": round(latencies[int(n * 0.95)], 0),
            "p99": round(latencies[int(n * 0.99)], 0),
        },
        "tokens": {
            "total_prompt": total_in,
            "total_completion": total_out,
            "mean_prompt_per_turn": round(total_in / n, 0) if n else 0,
            "mean_completion_per_turn": round(total_out / n, 0) if n else 0,
        },
        "cost_rmb": {
            "total": round(cost, 4),
            "per_turn": round(cost / n, 6) if n else 0,
            "price_per_1m_input": PRICE_PER_1M["input"],
            "price_per_1m_output": PRICE_PER_1M["output"],
        },
        "by_difficulty": _by("difficulty"),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    configure_logging(level="WARNING", json=False)
    settings = get_settings()

    # Minimal in-process checkpointer (no Redis needed just for eval)
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_graph(MemorySaver())
    llm_caller = LLMCaller(settings.llm)
    embedder = get_embedding(settings.llm, settings.embedding)

    items = load_qa()
    if args.limit:
        items = items[: args.limit]

    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded(item, idx):
        async with sem:
            return await _run_one(
                graph,
                embedder,
                llm_caller,
                settings,
                f"eval-sys-{idx}",
                item,
            )

    print(f"running system eval on {len(items)} queries (concurrency={CONCURRENCY}) …")
    t0 = time.perf_counter()
    rows = await asyncio.gather(*[bounded(it, i) for i, it in enumerate(items)])
    wall = time.perf_counter() - t0

    report = _summarize(list(rows))
    report["wall_s"] = round(wall, 1)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lat = report["latency_ms"]
    tok = report["tokens"]
    cost = report["cost_rmb"]
    print(f"\n=== system eval — {report['n']} turns (wall {wall:.0f}s) ===")
    lat_s = f"mean={lat['mean']:.0f}ms  median={lat['median']:.0f}ms  p95={lat['p95']:.0f}ms"
    tok_s = f"{tok['mean_prompt_per_turn']:.0f} in + {tok['mean_completion_per_turn']:.0f} out"
    print(f"latency: {lat_s}")
    print(f"tokens:  {tok_s} per turn")
    print(f"cost:    ¥{cost['total']:.4f} total  ¥{cost['per_turn']:.6f}/turn")
    print("\nby difficulty:")
    for diff, m in report["by_difficulty"].items():
        print(f"  {diff:<8}: mean={m['mean_ms']:.0f}ms  p95={m['p95_ms']:.0f}ms")
    print(f"\nreport -> {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
