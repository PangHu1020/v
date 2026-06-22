"""Run conversational retrieval eval (stage 4).

For each case: loop [user_sim generates utterance → agent graph processes it →
collect tool messages] until the user simulator says ``done`` or ``max_turns``
is reached. Extract all retrieved product ids from the tool messages, score
against the gold set (any-of), and report hit rate + turn efficiency.

Usage::

    uv run python -m backend.eval.conversational.run_conv_eval
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from backend.eval.common import TokenCounter, get_embedder, get_llm_caller, load_products
from backend.eval.conversational.cases import ConvResult, load_conv_cases
from backend.eval.conversational.judge import judge_task_completion
from backend.eval.conversational.ragas_score import (
    build_reference,
    extract_contexts,
    scorable,
    score_cases,
)
from backend.eval.conversational.scoring import extract_retrieved_ids, score_hit
from backend.eval.conversational.user_sim import simulate_user
from backend.v.agents.graph import GRAPH_RECURSION_LIMIT, build_graph
from backend.v.configs import get_settings
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("eval.run_conv")

CONCURRENCY = 3  # parallel cases (each case runs its own multi-turn dialogue)


async def _eval_one(
    graph, llm, *, case, thread_id: str, session_id: str, embedder=None, settings=None
) -> ConvResult:
    """Run one case: multi-turn dialogue until done or max_turns."""
    transcript: list[dict[str, str]] = []
    tool_outputs: list[str] = []
    hit_turn = 0

    counter = TokenCounter()
    config = {
        "recursion_limit": GRAPH_RECURSION_LIMIT,
        "configurable": {
            "thread_id": thread_id,
            "llm_caller": llm,
            # search tool reads these from config — REQUIRED (no real fallback;
            # missing embedder makes search return "缺少运行上下文" → agent
            # paraphrases as a system error and hallucinates products).
            "embedder": embedder,
            "settings": settings,
        },
        "callbacks": [counter],  # tallies tokens across every LLM call this case
    }
    t0 = time.perf_counter()

    for turn in range(1, case.max_turns + 1):
        # User simulator generates the next customer utterance.
        user_msg, done = await simulate_user(llm, case=case, transcript=transcript)
        transcript.append({"role": "user", "content": user_msg})

        # Agent graph processes the message.
        state_in = {
            "messages": [HumanMessage(content=user_msg)],
            "session_id": session_id,
            "channel": "eval",
            "channel_user_id": f"user-{case.case_id}",
            "user_profile": {},
        }
        try:
            final = await graph.ainvoke(state_in, config)
        except Exception as exc:
            _log.warning("eval.case_failed", case_id=case.case_id, turn=turn, error=str(exc))
            break

        # Collect agent's final text response (last AIMessage).
        agent_reply = ""
        for msg in reversed(final.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                agent_reply = msg.content
                break
        if agent_reply:
            transcript.append({"role": "agent", "content": agent_reply})

        # Collect tool outputs (all ToolMessages in this turn).
        for msg in final.get("messages", []):
            if isinstance(msg, ToolMessage) and msg.content:
                tool_outputs.append(msg.content)

        # Check if we hit gold this turn (first hit wins).
        if hit_turn == 0:
            retrieved = extract_retrieved_ids(tool_outputs, source_type="product")
            if score_hit(retrieved, case.gold_source_ids):
                hit_turn = turn

        if done:
            break

    # Final scoring.
    latency_ms = (time.perf_counter() - t0) * 1000
    all_retrieved = extract_retrieved_ids(tool_outputs, source_type="product")
    hit = score_hit(all_retrieved, case.gold_source_ids)
    tool_calls = len(tool_outputs)
    tool_errors = sum(
        1 for o in tool_outputs if o.lstrip().startswith(("[tool_error]", "[tool_guard]"))
    )
    completed, reason = await judge_task_completion(llm, case=case, transcript=transcript)

    # Capture RAGAS generation-quality inputs (scored post-hoc).
    final_response = ""
    for t in reversed(transcript):
        if t["role"] == "agent" and t["content"]:
            final_response = t["content"]
            break
    retrieved_contexts = extract_contexts(tool_outputs)

    return ConvResult(
        case_id=case.case_id,
        persona=case.persona,
        category=case.constraint["category"],
        gold_source_ids=case.gold_source_ids,
        retrieved_source_ids=all_retrieved,
        hit=hit,
        hit_turn=hit_turn,
        total_turns=len([t for t in transcript if t["role"] == "user"]),
        task_completed=completed,
        task_reason=reason,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        prompt_tokens=counter.prompt_tokens,
        completion_tokens=counter.completion_tokens,
        latency_ms=latency_ms,
        transcript=transcript,
        retrieved_contexts=retrieved_contexts,
        final_response=final_response,
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases", type=Path, help="conv_cases.jsonl path (default: data/conv_cases.jsonl)"
    )
    parser.add_argument("--out", type=Path, help="output report json (default: stdout)")
    parser.add_argument("--limit", type=int, help="only eval first N cases (for quick testing)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY,
        help=f"parallel cases (default {CONCURRENCY}); set 1 to serialize and avoid "
        "rate-limiting on hosted APIs.",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="skip the RAGAS generation-quality stage (retrieval/feedback metrics only).",
    )
    parser.add_argument(
        "--judge-base-url",
        default=os.environ.get("EVAL_JUDGE_BASE_URL", ""),
        help="OpenAI-compatible base_url for the RAGAS judge LLM "
        "(default: $EVAL_JUDGE_BASE_URL, else the agent's own endpoint).",
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.environ.get("EVAL_JUDGE_API_KEY", ""),
        help="API key for the judge endpoint (default: $EVAL_JUDGE_API_KEY).",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("EVAL_JUDGE_MODEL", ""),
        help="Model name for the judge (default: $EVAL_JUDGE_MODEL).",
    )
    args = parser.parse_args()

    configure_logging(level="WARNING", json=False)
    cases = load_conv_cases(args.cases) if args.cases else load_conv_cases()
    if args.limit:
        cases = cases[: args.limit]

    _log.info("eval.start", total_cases=len(cases))
    llm = get_llm_caller()
    embedder = get_embedder()
    settings = get_settings()
    ckpt = MemorySaver()  # in-memory checkpointer (eval doesn't need Redis)
    graph = build_graph(ckpt)

    # Run cases concurrently.
    sem = asyncio.Semaphore(args.concurrency)

    async def _wrapped(idx, c):
        async with sem:
            return await _eval_one(
                graph,
                llm,
                case=c,
                thread_id=f"eval-{c.case_id}",
                session_id=f"sess-{c.case_id}",
                embedder=embedder,
                settings=settings,
            )

    tasks = [_wrapped(i, c) for i, c in enumerate(cases)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Separate valid results from exceptions. Crucially, surface WHY cases
    # failed — a silent gather(return_exceptions=True) would otherwise report
    # "0 cases" indistinguishably whether the retriever missed everything or
    # the LLM API was down (e.g. account arrears returns 400 on every call).
    valid = [r for r in results if isinstance(r, ConvResult)]
    failures = [r for r in results if isinstance(r, BaseException)]
    if failures:
        from collections import Counter

        kinds = Counter(type(e).__name__ for e in failures)
        _log.warning("eval.cases_failed", count=len(failures), kinds=dict(kinds))
        # Print to stdout too so it's visible without log scraping.
        print(f"\n⚠️  {len(failures)}/{len(results)} cases FAILED (not retrieval misses):")
        for kind, n in kinds.most_common():
            sample = next(e for e in failures if type(e).__name__ == kind)
            print(f"    {kind} ×{n}: {str(sample)[:160]}")
        if not valid:
            print(
                "\n  All cases failed — this is an infrastructure/API error, "
                "NOT a 0% hit rate. Fix the cause above and re-run."
            )
    _log.info("eval.done", valid=len(valid), failed=len(failures))

    # ── RAGAS generation-quality stage (post-hoc, independent judge) ────────
    ragas_aggregate: dict[str, float] = {}
    ragas_scored = 0
    if not args.no_ragas and valid:
        judge_url = args.judge_base_url or settings.llm.base_url
        judge_key = args.judge_api_key or settings.llm.api_key
        judge_model = args.judge_model or settings.llm.model
        products = {p.product_id: p for p in load_products()}
        samples = [
            {
                "case_id": r.case_id,
                "user_input": next((c.hidden_need for c in cases if c.case_id == r.case_id), ""),
                "response": r.final_response,
                "retrieved_contexts": r.retrieved_contexts,
                "reference": build_reference(r.gold_source_ids, products),
            }
            for r in valid
        ]
        ragas_scored = sum(1 for s in samples if scorable(s))
        print(f"\nrunning RAGAS … (judge={judge_model}, scorable {ragas_scored}/{len(samples)})")
        t_ragas = time.perf_counter()
        try:
            ragas_aggregate, per_case = await score_cases(
                samples,
                judge_base_url=judge_url,
                judge_api_key=judge_key,
                judge_model=judge_model,
                embed_base_url=settings.embedding.base_url,
                embed_api_key=settings.embedding.api_key,
                embed_model=settings.embedding.model,
                concurrency=args.concurrency,
            )
            # Fold per-case scores back onto the results for the JSON dump.
            by_id = {pc["case_id"]: pc for pc in per_case}
            for r in valid:
                pc = by_id.get(r.case_id, {})
                r.ragas = {k: v for k, v in pc.items() if k != "case_id"}
            print(f"  RAGAS done in {time.perf_counter() - t_ragas:.1f}s")
        except Exception as exc:
            _log.warning("eval.ragas_failed", error=type(exc).__name__, detail=str(exc)[:200])
            print(f"  ⚠️  RAGAS stage failed ({type(exc).__name__}); reporting without it.")

    # Aggregate.
    n = len(valid)
    hit_count = sum(1 for r in valid if r.hit)
    hit_rate = hit_count / n if valid else 0.0
    avg_turns = sum(r.total_turns for r in valid) / n if valid else 0.0
    avg_hit_turn = (
        sum(r.hit_turn for r in valid if r.hit_turn > 0) / hit_count if hit_count else 0.0
    )
    # Feedback-loop metrics.
    completed_count = sum(1 for r in valid if r.task_completed)
    task_rate = completed_count / n if valid else 0.0
    total_tool_calls = sum(r.tool_calls for r in valid)
    total_tool_errors = sum(r.tool_errors for r in valid)
    tool_success = (
        (total_tool_calls - total_tool_errors) / total_tool_calls if total_tool_calls else 1.0
    )
    avg_total_tokens = (
        sum(r.prompt_tokens + r.completion_tokens for r in valid) / n if valid else 0.0
    )
    latencies = sorted(r.latency_ms for r in valid)
    p50 = latencies[n // 2] if valid else 0.0
    p95 = latencies[int(n * 0.95)] if valid else 0.0

    # Retrieval depth metrics (beyond binary any-of hit).
    searched = [r for r in valid if r.tool_calls > 0]
    searched_hit = sum(1 for r in searched if r.hit)
    searched_hit_rate = searched_hit / len(searched) if searched else 0.0
    no_search = n - len(searched)
    # Gold coverage: fraction of each case's gold set that was retrieved, averaged.
    gold_cov = (
        sum(
            len(set(r.retrieved_source_ids) & set(r.gold_source_ids)) / len(r.gold_source_ids)
            for r in valid
            if r.gold_source_ids
        )
        / n
        if valid
        else 0.0
    )
    avg_prompt = sum(r.prompt_tokens for r in valid) / n if valid else 0.0
    avg_completion = sum(r.completion_tokens for r in valid) / n if valid else 0.0

    # By persona / category.
    from collections import Counter

    by_persona = Counter(r.persona for r in valid)
    by_cat = Counter(r.category for r in valid)
    persona_hits = {p: sum(1 for r in valid if r.persona == p and r.hit) for p in by_persona}
    cat_hits = {c: sum(1 for r in valid if r.category == c and r.hit) for c in by_cat}
    persona_done = {
        p: sum(1 for r in valid if r.persona == p and r.task_completed) for p in by_persona
    }

    print(f"\n{'=' * 60}")
    print(f"Conversational Eval — {len(valid)} cases")
    print(f"{'=' * 60}")
    print("── Retrieval ──")
    print(f"  Hit rate (any-of, all)     : {hit_rate:.1%} ({hit_count}/{n})")
    print(
        f"  Hit rate (cases searched)  : {searched_hit_rate:.1%} ({searched_hit}/{len(searched)})"
    )
    print(f"  Avg gold coverage          : {gold_cov:.1%}")
    print(f"  Cases agent never searched : {no_search}/{n}")
    print(f"  Avg turns / first-hit turn : {avg_turns:.1f} / {avg_hit_turn:.1f}")
    print("── Generation (RAGAS) ──")
    if ragas_aggregate:
        print(f"  (judge-scored on {ragas_scored}/{n} cases with response+contexts)")
        for k in ("Faithfulness", "AnswerRelevancy", "ContextRecall", "AnswerCorrectness"):
            if k in ragas_aggregate:
                print(f"  {k:<25}: {ragas_aggregate[k]:.4f}")
    else:
        print("  (skipped — use --judge-* / EVAL_JUDGE_* to enable)")
    print("── Task / Tools / Cost ──")
    tool_ok = total_tool_calls - total_tool_errors
    print(f"  Task completion (judge)    : {task_rate:.1%} ({completed_count}/{n})")
    print(f"  Tool-call success rate     : {tool_success:.1%} ({tool_ok}/{total_tool_calls} calls)")
    print(
        f"  Avg tokens (prompt/compl)  : {avg_total_tokens:.0f} "
        f"({avg_prompt:.0f}/{avg_completion:.0f})"
    )
    print(f"  Latency p50 / p95 (case)   : {p50:.0f}ms / {p95:.0f}ms")
    print("\nBy persona (hit / task-done):")
    for p in sorted(by_persona):
        t = by_persona[p]
        print(f"  {p:15s}: {persona_hits[p]}/{t} hit, {persona_done[p]}/{t} done")
    print("\nBy category (hit):")
    for c in sorted(cat_hits):
        t = by_cat[c]
        print(f"  {c:10s}: {cat_hits[c]}/{t} = {cat_hits[c] / t:.1%}")

    # Dump misses for debug.
    misses = [r for r in valid if not r.hit]
    if misses:
        print(f"\n{'=' * 60}")
        print(f"Missed cases ({len(misses)}):")
        for r in misses[:5]:  # show first 5
            print(f"\n[{r.case_id}] {r.persona} / {r.category} (gold={len(r.gold_source_ids)})")
            print(f"  retrieved: {r.retrieved_source_ids}")
            print(f"  transcript ({r.total_turns} turns):")
            for t in r.transcript[:6]:  # first 3 exchanges
                role_label = "客户" if t["role"] == "user" else "客服"
                content = t["content"][:80] + "..." if len(t["content"]) > 80 else t["content"]
                print(f"    {role_label}: {content}")

    # Optionally write full results.
    if args.out:
        import json

        report = {
            "n_cases": n,
            "retrieval": {
                "hit_rate": round(hit_rate, 4),
                "hit_count": hit_count,
                "hit_rate_searched": round(searched_hit_rate, 4),
                "cases_searched": len(searched),
                "cases_no_search": no_search,
                "avg_gold_coverage": round(gold_cov, 4),
                "avg_turns": round(avg_turns, 2),
                "avg_first_hit_turn": round(avg_hit_turn, 2),
                "by_category": {c: {"hit": cat_hits[c], "total": by_cat[c]} for c in by_cat},
                "by_persona": {
                    p: {"hit": persona_hits[p], "total": by_persona[p]} for p in by_persona
                },
            },
            "ragas": ragas_aggregate,
            "ragas_scored_cases": ragas_scored,
            "task": {
                "completion_rate": round(task_rate, 4),
                "completed": completed_count,
            },
            "tools": {
                "success_rate": round(tool_success, 4),
                "calls": total_tool_calls,
                "errors": total_tool_errors,
            },
            "cost": {
                "avg_prompt_tokens": round(avg_prompt, 1),
                "avg_completion_tokens": round(avg_completion, 1),
                "avg_total_tokens": round(avg_total_tokens, 1),
            },
            "latency_ms": {"p50": round(p50, 1), "p95": round(p95, 1)},
            "cases": [r.to_dict() for r in valid],
        }
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nFull report → {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
