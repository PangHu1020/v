"""Generate conversational retrieval eval cases (Route C, stage 2).

Pipeline:
1. For each category, derive price bands by grouping sorted products into
   windows of 2-7 items → each band is a machine-checkable constraint with a
   complete, non-trivial gold set (enumerated by ``enumerate_gold``).
2. Assign a persona round-robin (stable, balanced distribution).
3. LLM generates the customer's ``hidden_need`` + ``noise`` for that constraint
   + persona — colloquial, emotional, info-dripping, and crucially NOT naming
   any specific product (else the case degrades to a single-gold lookup).

Writes ``data/conv_cases.jsonl``. Re-runnable; the committed file is frozen.
The noise/emotion *variance* is dialled up in the prompt so the dataset spans
mild → harsh, giving the eval discriminative power (per user feedback that a
uniformly "中规中矩" set is too flat).

Usage::

    uv run python -m backend.eval.conversational.gen_conv_cases
"""

from __future__ import annotations

import argparse
import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from backend.eval.common import (
    CATEGORIES,
    Product,
    dump_jsonl,
    get_llm_caller,
    load_products,
)
from backend.eval.conversational.cases import (
    CONV_CASES_PATH,
    PERSONAS,
    ConvCase,
    enumerate_gold,
)
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("eval.gen_conv")

CONCURRENCY = 6
MIN_GOLD = 2  # a case must have ≥2 gold products (else it's single-gold)
MAX_GOLD = 7  # keep bands focused; very broad bands aren't useful queries
BAND_TARGET = 4  # aim ~4 products per band when slicing

_PERSONA_GUIDE = {
    "budget_tight": "预算卡死，反复强调别太贵、要比价，对价格敏感。",
    "in_a_hurry": "急着要，没耐心，回答简短，信息要客服一点点追问才挤出来。",
    "clueless": "不懂参数，说一堆生活场景细节，需要客服引导，问得很模糊。",
    "frustrated": "带怨气（上次买贵了/被坑过），强调这次必须值，语气冲。",
    "gifting": "给别人买（对象/家人），自己不懂，需求飘忽，可能顺嘴问活动。",
    "terse": "惜字如金，经常一两个字一句，全程被动，全靠客服追问。",
    "rambler": "话痨跑题严重，先扯一堆闲事（天气/吐槽），正经诉求埋在废话里。",
    "wishy_washy": "边聊边改主意，预算或需求前后变化，让客服跟着调整。",
}


class _NeedGen(BaseModel):
    """LLM output: the customer's hidden need + noise for one case."""

    hidden_need: str = Field(
        description="客户的真实诉求，一句话白描（如'想要预算两三千、性能好的安卓手机'）。"
        "不要点名任何具体商品型号。"
    )
    noise: str = Field(
        description="客户会夹带的闲聊/情绪/跑题话题（如'抱怨上次买贵了''先扯几句天气'）。"
        "可以为空字符串表示这个case噪声较少。"
    )


_SYS = (
    "你在为'对话式商品检索'评测造数据。给定一个商品筛选约束（类目+价格区间）和一种客户画像，"
    "想象一个真实客户带着这个需求来咨询客服，"
    "输出他的【真实诉求 hidden_need】和【会夹带的噪声 noise】。\n"
    "要求：\n"
    "1. hidden_need 用大白话描述需求，体现约束（类目+大致预算），但【绝对不要】点名任何具体商品型号"
    "（不能出现'iPhone 15'这种），否则评测会退化成查字典。\n"
    "2. noise 体现该画像的说话特征——口语化、情绪化、可跑题。画像越极端，noise 越浓。\n"
    "3. 只输出 JSON：hidden_need、noise。"
)


def _derive_constraints(products: list[Product]) -> list[dict]:
    """Slice each category's price range into bands yielding MIN_GOLD..MAX_GOLD products."""
    constraints: list[dict] = []
    for cat in CATEGORIES:
        items = sorted((p for p in products if p.category == cat), key=lambda p: p.price)
        if len(items) < MIN_GOLD:
            continue
        # Slide a window of BAND_TARGET items; band = [first.price, last.price].
        i = 0
        while i < len(items):
            window = items[i : i + BAND_TARGET]
            if len(window) < MIN_GOLD:
                break
            lo = window[0].price
            hi = window[-1].price
            constraint = {"category": cat, "price_min": lo, "price_max": hi}
            gold = enumerate_gold(products, constraint)
            if MIN_GOLD <= len(gold) <= MAX_GOLD:
                constraints.append(constraint)
            i += BAND_TARGET
    return constraints


async def _gen_one(
    llm,
    sem: asyncio.Semaphore,
    *,
    case_id: str,
    constraint: dict,
    gold: list[str],
    persona: str,
) -> ConvCase | None:
    lo = constraint["price_min"]
    hi = constraint["price_max"]
    prompt = [
        SystemMessage(content=_SYS),
        HumanMessage(
            content=(
                f"<约束>类目:{constraint['category']} 价格区间:{lo:g}-{hi:g}元</约束>\n"
                f"<客户画像>{persona}：{_PERSONA_GUIDE[persona]}</客户画像>"
            )
        ),
    ]
    try:
        async with sem:
            res = await llm.chat("main_primary", prompt, structured=_NeedGen)
        g: _NeedGen = res.parsed
        if not g or not g.hidden_need.strip():
            return None
        return ConvCase(
            case_id=case_id,
            constraint=constraint,
            gold_source_ids=gold,
            persona=persona,
            hidden_need=g.hidden_need.strip(),
            noise=g.noise.strip(),
        )
    except Exception as exc:
        _log.warning("gen_conv.case_failed", case_id=case_id, error=type(exc).__name__)
        return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-constraint", type=int, default=1, help="cases per constraint (different personas)"
    )
    args = parser.parse_args()

    configure_logging(level="WARNING", json=False)
    products = load_products()
    if not products:
        raise SystemExit("no products.jsonl — run expand_catalog first")

    constraints = _derive_constraints(products)
    _log.info("gen_conv.constraints", count=len(constraints))

    # Build tasks: each constraint × per-constraint personas (round-robin).
    llm = get_llm_caller()
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = []
    idx = 0
    for constraint in constraints:
        gold = enumerate_gold(products, constraint)
        for _ in range(args.per_constraint):
            persona = PERSONAS[idx % len(PERSONAS)]
            case_id = f"CONV{idx + 1:04d}"
            tasks.append(
                _gen_one(
                    llm, sem, case_id=case_id, constraint=constraint, gold=gold, persona=persona
                )
            )
            idx += 1

    results = await asyncio.gather(*tasks)
    cases = [c for c in results if c is not None]
    dump_jsonl(CONV_CASES_PATH, [c.to_dict() for c in cases])
    print(f"generated {len(cases)} conv cases -> {CONV_CASES_PATH}")
    # Quick distribution summary.
    from collections import Counter

    by_persona = Counter(c.persona for c in cases)
    by_cat = Counter(c.constraint["category"] for c in cases)
    gold_sizes = [len(c.gold_source_ids) for c in cases]
    avg_gold = sum(gold_sizes) / len(gold_sizes)
    print(f"  personas: {dict(by_persona)}")
    print(f"  categories: {dict(by_cat)}")
    print(f"  gold size: min={min(gold_sizes)} max={max(gold_sizes)} avg={avg_gold:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
