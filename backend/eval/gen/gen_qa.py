"""LLM QA generator: products + FAQ -> 300+ tiered retrieval-eval items.

Each QA item is **anchored to exactly one source document** (one product or one
FAQ entry). The anchor's ``source_id`` becomes the gold label. The LLM is asked
to write a query that is answerable from that document alone, in one of five
difficulty tiers:

- ``plain``          — direct, well-formed lookup ("iPhone 15 Pro 多少钱").
- ``colloquial``     — chatty / spoken, filler words ("那个苹果的新手机大概啥价位啊").
- ``synonym``        — paraphrase, no surface term overlap ("苹果出的高端机型贵不贵").
- ``noisy``          — emotional / complaint framing around the real ask.
- ``multi_condition``— brand + attribute + intent stacked in one sentence.

The noisy + synonym + multi_condition tiers are what actually exercise the
retriever's stage-2 (hybrid) and stage-3 (LLM-rewrite) cascade.

Writes ``data/qa.jsonl``. Re-runnable; the committed file is the frozen eval set.

Usage::

    uv run python -m backend.eval.gen.gen_qa
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from backend.eval.common import (
    QA_PATH,
    TIERS,
    FaqEntry,
    Product,
    QaItem,
    build_faq_text,
    build_product_text,
    dump_jsonl,
    get_llm_caller,
    load_faqs,
    load_products,
)
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("eval.gen_qa")

QA_PER_PRODUCT = 2  # 120 products * 2 = 240
QA_PER_FAQ = 3  # 30 faq * 3 = 90  -> 330 tasks total, margin above the 300+ target
CONCURRENCY = 8

_TIER_GUIDE = {
    "plain": "直接、规范的查询，包含商品名或核心属性。例：iPhone 15 Pro 多少钱",
    "colloquial": "口语化、带语气词和废话的问法。例：那个苹果新出的pro大概要多少钱啊",
    "synonym": "用同义/近义改写，尽量不与文档原词面重叠。例：苹果的高端旗舰机贵不贵",
    "noisy": "带情绪或抱怨的包裹，但核心诉求明确。例：催了半天到底这空调多少钱能不能说清楚",
    "multi_condition": "把品牌+属性+诉求叠加在一句里。例：想买台美的的变频空调预算三千左右有没有",
}

_SYS = """你是检索评测数据集的出题人。给定一条商品或FAQ文档，请按指定难度风格写出顾客可能\
发出的查询（query），并给出该查询的简短参考答案（answer，依据文档内容，20-50字）。\
要求：\
1. query 必须能**仅凭这一条文档**回答，不要引入文档里没有的其他商品/政策。\
2. 按 tier 指定的风格写 query，风格差异要明显。\
3. answer 用中文，简洁准确，基于文档事实。\
只输出 JSON。"""


class _QaGen(BaseModel):
    query: str = Field(description="顾客查询")
    answer: str = Field(description="基于文档的参考答案")


async def _gen_for_doc(
    llm,
    sem: asyncio.Semaphore,
    *,
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
            answer=g.answer.strip(),
        )
    except Exception as exc:
        _log.warning("gen_qa.doc_failed", source_id=source_id, error=type(exc).__name__)
        return None


def _plan(products: list[Product], faqs: list[FaqEntry]) -> list[tuple[str, str, str, str]]:
    """Build (source_type, source_id, doc_text, tier) tasks.

    Tiers are assigned round-robin by a stable index so the committed dataset
    has a balanced, reproducible tier distribution.
    """
    tasks: list[tuple[str, str, str, str]] = []
    idx = 0
    for p in products:
        text = build_product_text(p)
        for _ in range(QA_PER_PRODUCT):
            tasks.append(("product", p.product_id, text, TIERS[idx % len(TIERS)]))
            idx += 1
    for f in faqs:
        text = build_faq_text(f)
        for _ in range(QA_PER_FAQ):
            tasks.append(("faq", f.faq_id, text, TIERS[idx % len(TIERS)]))
            idx += 1
    return tasks


async def main() -> None:
    configure_logging(level="INFO", json=False)
    products = load_products()
    faqs = load_faqs()
    if not products:
        raise SystemExit("no products.jsonl — run expand_catalog first")

    llm = get_llm_caller()
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = _plan(products, faqs)

    coros = [
        _gen_for_doc(
            llm,
            sem,
            source_type=st,
            source_id=sid,
            doc_text=text,
            tier=tier,
            qa_id=f"QA{i:04d}",
        )
        for i, (st, sid, text, tier) in enumerate(tasks, start=1)
    ]
    results = await asyncio.gather(*coros)
    items = [r for r in results if r is not None]

    dump_jsonl(QA_PATH, [it.to_dict() for it in items])
    by_tier: dict[str, int] = {}
    for it in items:
        by_tier[it.tier] = by_tier.get(it.tier, 0) + 1
    _log.info("gen_qa.complete", total=len(items), by_tier=by_tier)
    print(f"wrote {len(items)} qa items -> {QA_PATH}")
    print("by tier:", by_tier)


if __name__ == "__main__":
    asyncio.run(main())
