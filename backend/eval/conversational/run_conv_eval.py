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
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from backend.eval.common import get_embedder, get_llm_caller
from backend.eval.conversational.cases import ConvResult, load_conv_cases
from backend.eval.conversational.scoring import extract_retrieved_ids, score_hit
from backend.eval.conversational.user_sim import simulate_user
from backend.v.agents.graph import build_graph
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

    config = {
        "configurable": {
            "thread_id": thread_id,
            "llm_caller": llm,
            # search tool reads these from config — REQUIRED (no real fallback;
            # missing embedder makes search return "缺少运行上下文" → agent
            # paraphrases as a system error and hallucinates products).
            "embedder": embedder,
            "settings": settings,
        }
    }

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
    all_retrieved = extract_retrieved_ids(tool_outputs, source_type="product")
    hit = score_hit(all_retrieved, case.gold_source_ids)
    return ConvResult(
        case_id=case.case_id,
        persona=case.persona,
        category=case.constraint["category"],
        gold_source_ids=case.gold_source_ids,
        retrieved_source_ids=all_retrieved,
        hit=hit,
        hit_turn=hit_turn,
        total_turns=len([t for t in transcript if t["role"] == "user"]),
        transcript=transcript,
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases", type=Path, help="conv_cases.jsonl path (default: data/conv_cases.jsonl)"
    )
    parser.add_argument("--out", type=Path, help="output report json (default: stdout)")
    parser.add_argument("--limit", type=int, help="only eval first N cases (for quick testing)")
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
    sem = asyncio.Semaphore(CONCURRENCY)

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

    # Filter exceptions.
    valid = [r for r in results if isinstance(r, ConvResult)]
    _log.info("eval.done", valid=len(valid), failed=len(results) - len(valid))

    # Aggregate.
    hit_count = sum(1 for r in valid if r.hit)
    hit_rate = hit_count / len(valid) if valid else 0.0
    avg_turns = sum(r.total_turns for r in valid) / len(valid) if valid else 0.0
    avg_hit_turn = (
        sum(r.hit_turn for r in valid if r.hit_turn > 0) / hit_count if hit_count else 0.0
    )

    # By persona / category.
    from collections import Counter

    by_persona = Counter(r.persona for r in valid)
    by_cat = Counter(r.category for r in valid)
    persona_hits = {p: sum(1 for r in valid if r.persona == p and r.hit) for p in by_persona}
    cat_hits = {c: sum(1 for r in valid if r.category == c and r.hit) for c in by_cat}

    print(f"\n{'=' * 60}")
    print(f"Conversational Retrieval Eval — {len(valid)} cases")
    print(f"{'=' * 60}")
    print(f"Hit rate (any-of): {hit_rate:.1%} ({hit_count}/{len(valid)})")
    print(f"Avg turns per case: {avg_turns:.1f}")
    print(f"Avg hit turn (when hit): {avg_hit_turn:.1f}")
    print("\nBy persona:")
    for p in sorted(persona_hits):
        p_hit = persona_hits[p]
        p_total = by_persona[p]
        print(f"  {p:15s}: {p_hit}/{p_total} = {p_hit / p_total:.1%}")
    print("\nBy category:")
    for c in sorted(cat_hits):
        c_hit = cat_hits[c]
        c_total = by_cat[c]
        print(f"  {c:10s}: {c_hit}/{c_total} = {c_hit / c_total:.1%}")

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

        args.out.write_text(
            json.dumps([r.to_dict() for r in valid], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nFull results → {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
