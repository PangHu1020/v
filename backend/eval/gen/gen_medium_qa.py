"""Generate additional medium-difficulty QA to balance the dataset.

Medium = requires synonym reasoning or colloquial rephrase, but answerable.
Uses synonym + colloquial tier styles (50/50), balanced product/faq sources.

Usage::

    uv run python -m backend.eval.gen.gen_medium_qa --count 80
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from backend.eval.common import (
    QA_PATH,
    QaItem,
    build_faq_text,
    build_product_text,
    get_llm_caller,
    load_faqs,
    load_products,
    load_qa,
)
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("eval.gen_medium_qa")
CONCURRENCY = 8

_TIERS = ["synonym", "colloquial"]
_TIER_GUIDE = {
    "synonym": "用同义/近义改写，尽量不与文档原词面重叠。例：苹果的高端旗舰机贵不贵",
    "colloquial": "口语化、带语气词，但核心意图清晰。例：那个苹果新出的pro大概要多少钱啊",
}

_SYS = (
    "你是检索评测数据集的出题人。给定一条商品或FAQ文档，请按指定风格写一条查询"
    "（query）和参考答案（answer）。要求：query 必须能仅凭这一条文档回答；"
    "按 tier 风格写 query；answer 依据文档事实，20-50字。只输出 JSON。"
)


class _QaGen(BaseModel):
    query: str
    answer: str


async def _gen(
    llm,
    sem: asyncio.Semaphore,
    source_type: str,
    source_id: str,
    doc_text: str,
    tier: str,
    qa_id: str,
) -> QaItem | None:
    prompt = [
        SystemMessage(content=_SYS),
        HumanMessage(
            content=(
                f"<tier>{tier}</tier>\n"
                f"<tier_style>{_TIER_GUIDE[tier]}</tier_style>\n"
                f"<document>\n{doc_text}\n</document>"
            )
        ),
    ]
    try:
        async with sem:
            res = await llm.chat("main_primary", prompt, structured=_QaGen)
        g: _QaGen = res.parsed
        if not g or not g.query.strip():
            return None
        return QaItem(
            qa_id=qa_id,
            query=g.query.strip(),
            gold_source_type=source_type,
            gold_source_ids=[source_id],
            tier=tier,
            difficulty="medium",
            answer=g.answer.strip(),
        )
    except Exception as exc:
        _log.warning("gen_medium.failed", source_id=source_id, error=type(exc).__name__)
        return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=80)
    args = parser.parse_args()

    configure_logging(level="WARNING", json=False)
    products = load_products()
    faqs = load_faqs()
    existing = load_qa()
    next_id = max(int(q.qa_id.replace("QA", "")) for q in existing) + 1

    # Build candidate pool: product + faq sources, balanced
    rng = random.SystemRandom()
    rng.seed = lambda *_: None
    prod_pool = [(p.product_id, "product", build_product_text(p)) for p in products]
    faq_pool = [(f.faq_id, "faq", build_faq_text(f)) for f in faqs]
    half = args.count // 2
    prod_sample = rng.choices(prod_pool, k=half)
    faq_sample = rng.choices(faq_pool, k=args.count - half)
    candidates = prod_sample + faq_sample
    rng.shuffle(candidates)

    llm = get_llm_caller()
    sem = asyncio.Semaphore(CONCURRENCY)
    tiers = [_TIERS[i % 2] for i in range(len(candidates))]

    coros = [
        _gen(
            llm,
            sem,
            src_type,
            src_id,
            text,
            tier,
            f"QA{next_id + i:04d}",
        )
        for i, ((src_id, src_type, text), tier) in enumerate(zip(candidates, tiers, strict=True))
    ]
    results = await asyncio.gather(*coros)
    new_items = [r for r in results if r is not None]

    # Append to qa.jsonl
    rows = [json.loads(line) for line in QA_PATH.open(encoding="utf-8")]
    rows.extend(it.to_dict() for it in new_items)
    QA_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(f"added {len(new_items)} medium QA items -> total {len(rows)}")


if __name__ == "__main__":
    asyncio.run(main())
