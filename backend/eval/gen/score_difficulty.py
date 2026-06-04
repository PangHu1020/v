"""LLM semantic difficulty scorer for QA items.

Scores each query on a 1-3 scale:
  1 (easy)   -- exact product name / brand keyword, dense search retrieves directly
  2 (medium) -- requires synonym reasoning, structural variation, or colloquial rephrase
  3 (hard)   -- heavy noise/emotion, no surface overlap, or multi-constraint combination

Usage::

    uv run python -m backend.eval.gen.score_difficulty
    uv run python -m backend.eval.gen.score_difficulty --limit 50  # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from backend.eval.common import QA_PATH, get_llm_caller, load_qa
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("eval.score_difficulty")

CONCURRENCY = 8

_SYS = """你是检索难度评估专家。给定一条电商客服查询，判断它对向量检索（embedding）的难度：

1（简单）：查询中直接包含商品名、品牌名或精确关键词，embedding 能轻松匹配文档。
  例："iPhone 15 Pro 多少钱"

2（中等）：需要同义词推理、口语化理解或结构改写才能匹配，embedding 可能找到但不一定排第一。
  例："苹果出的高端旗舰机贵不贵"、"那个戴森吸尘器大概多少"

3（困难）：查询含大量情绪噪音、多个约束叠加、或完全无关键词面覆盖，embedding 很可能找不到。
  例："催了半天了能不能说清楚你们这个空调到底什么价"（有噪音）
  例："想买台可以拍星空的手机预算六千内还得支持双卡"（多约束无精确名）

只输出数字 1、2 或 3，不要解释。"""


class _Score(BaseModel):
    score: int = Field(ge=1, le=3)


async def _score_one(
    llm,
    sem: asyncio.Semaphore,
    query: str,
) -> int:
    prompt = [
        SystemMessage(content=_SYS),
        HumanMessage(content=query),
    ]
    try:
        async with sem:
            res = await llm.chat("main_primary", prompt, structured=_Score)
        parsed = res.parsed
        if isinstance(parsed, _Score):
            return parsed.score
        # fallback: try to parse from message content
        raw = (res.message.content or "").strip()
        if raw in ("1", "2", "3"):
            return int(raw)
    except Exception as exc:
        _log.warning("score.failed", error=type(exc).__name__, query=query[:40])
    return 2  # safe default: medium


TIER_TO_DIFFICULTY = {"easy": 1, "medium": 2, "hard": 3}
SCORE_TO_LABEL = {1: "easy", 2: "medium", 3: "hard"}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    configure_logging(level="WARNING", json=False)
    items = load_qa()
    if args.limit:
        items = items[: args.limit]

    llm = get_llm_caller()
    sem = asyncio.Semaphore(CONCURRENCY)

    scores = await asyncio.gather(*[_score_one(llm, sem, it.query) for it in items])

    # Write back to qa.jsonl with difficulty field
    rows = [json.loads(line) for line in QA_PATH.open(encoding="utf-8")]
    # Only update the items we scored (respects --limit)
    id_to_score = {it.qa_id: s for it, s in zip(items, scores, strict=True)}
    for row in rows:
        if row["qa_id"] in id_to_score:
            row["difficulty"] = SCORE_TO_LABEL[id_to_score[row["qa_id"]]]

    QA_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    from collections import Counter

    dist = Counter(id_to_score[it.qa_id] for it in items)
    n = len(items)
    print(f"scored {n} queries")
    print(f"  easy   (1): {dist[1]:3d}  {dist[1] / n * 100:.1f}%")
    print(f"  medium (2): {dist[2]:3d}  {dist[2] / n * 100:.1f}%")
    print(f"  hard   (3): {dist[3]:3d}  {dist[3] / n * 100:.1f}%")
    print(f"\nwritten -> {QA_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
