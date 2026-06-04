"""LLM catalog expander: 15 seed products -> ~120 products + ~30 FAQ entries.

Two passes, both via DeepSeek structured output:

1. **Enrich** the 15 canonical seed products (from ``dw.dim_product``) in
   place — keep their exact ``product_id`` / ``product_name`` / ``category`` /
   ``brand``, add a realistic ``price``, a sales ``description``, and a handful
   of spec key/values.
2. **Invent** new products per category (ids continue from ``P016``) until each
   category reaches ``PER_CATEGORY`` items, then a batch of FAQ / policy
   entries (``FAQ001`` …).

Writes ``data/products.jsonl`` + ``data/faq.jsonl``. Re-runnable; output is the
frozen artifact the seeder + eval consume, so the LLM non-determinism is
captured once at generation time and committed.

Usage::

    uv run python -m backend.eval.gen.expand_catalog
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from backend.eval.common import (
    CATEGORIES,
    FAQ_PATH,
    PRODUCTS_PATH,
    FaqEntry,
    Product,
    dump_jsonl,
    get_llm_caller,
)
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger

_log = get_logger("eval.expand_catalog")

PER_CATEGORY = 20  # 6 categories * 20 = 120 products total
NEW_PER_CALL = 5  # products invented per LLM call (smaller = faster, avoids timeout)
FAQ_TARGET = 30
CONCURRENCY = 4

# The canonical seed catalog (verbatim from scripts/sql/020_schema_dw.sql).
SEED = [
    ("P001", "iPhone 15 Pro", "手机数码", "苹果"),
    ("P002", "Galaxy S24 Ultra", "手机数码", "三星"),
    ("P003", "Mate 60 Pro", "手机数码", "华为"),
    ("P004", "戴森 V15 吸尘器", "家用电器", "戴森"),
    ("P005", "美的空调 KFR-35GW", "家用电器", "美的"),
    ("P006", "耐克 Air Max 270 运动鞋", "鞋靴", "耐克"),
    ("P007", "阿迪达斯 Ultraboost 跑鞋", "鞋靴", "阿迪达斯"),
    ("P008", "优衣库 Heattech 保暖夹克", "服饰", "优衣库"),
    ("P009", "李维斯 501 牛仔裤", "服饰", "李维斯"),
    ("P010", "雀巢金牌速溶咖啡", "食品饮料", "雀巢"),
    ("P011", "蒙牛纯牛奶 250ml*12", "食品饮料", "蒙牛"),
    ("P012", "乐事原味薯片 150g", "休闲零食", "乐事"),
    ("P013", "奥利奥巧克力夹心饼干", "休闲零食", "奥利奥"),
    ("P014", "Kindle Paperwhite 电子书", "手机数码", "亚马逊"),
    ("P015", "Instant Pot 多功能电压力锅", "家用电器", "Instant Pot"),
]

FAQ_TOPICS = [
    "退货退款",
    "换货",
    "物流配送",
    "运费邮费",
    "会员权益",
    "积分规则",
    "保修售后",
    "发票开具",
    "支付方式",
    "优惠券使用",
]

_ENRICH_SYS = """你是电商商品资料编辑。给定若干商品的基础信息（编号/名称/品类/品牌），\
为每个商品补全：合理的人民币售价（price，数字，单位元）、一段 30-60 字的中文卖点描述\
（description）、以及 3-5 个关键规格键值对（specs，键和值都是简短中文/数字）。\
不要修改 product_id / product_name / category / brand，原样返回。\
售价要符合该品类该品牌的真实档位。"""

_INVENT_SYS = """你是电商选品编辑。请为指定品类虚构若干**真实感强、互不重复**的在售商品。\
每个商品需要：product_name（含品牌和型号的完整中文商品名）、brand（品牌）、\
price（人民币售价，数字）、description（30-60 字中文卖点）、specs（3-5 个关键规格键值对）。\
category 固定为给定品类。不要编造与已有商品重名的条目。product_id 留空，由程序分配。"""

_FAQ_SYS = """你是电商客服知识库编辑。请围绕给定主题各写一条 FAQ：topic（主题）、\
question（顾客常见问法，口语化）、answer（清晰、可执行的中文回答，40-100 字，\
包含具体规则如天数/金额/条件）。内容要符合中国电商平台的常见政策。"""


class _ProductGen(BaseModel):
    product_id: str = Field(default="", description="编号，可留空")
    product_name: str
    category: str
    brand: str
    price: float
    description: str
    specs: dict[str, str] = Field(default_factory=dict)


class _ProductBatch(BaseModel):
    products: list[_ProductGen]


class _FaqGen(BaseModel):
    topic: str
    question: str
    answer: str


class _FaqBatch(BaseModel):
    faqs: list[_FaqGen]


async def _enrich_seed(llm) -> list[Product]:
    payload = "\n".join(f"{pid} | {name} | {cat} | {brand}" for pid, name, cat, brand in SEED)
    prompt = [
        SystemMessage(content=_ENRICH_SYS),
        HumanMessage(content=f"<products>\n{payload}\n</products>"),
    ]
    res = await llm.chat("main_primary", prompt, structured=_ProductBatch)
    parsed: _ProductBatch = res.parsed
    by_id = {p.product_id: p for p in parsed.products}
    out: list[Product] = []
    for pid, name, cat, brand in SEED:
        g = by_id.get(pid)
        if g is None:
            # Fall back to a minimal record so the seed item is never dropped.
            out.append(Product(pid, name, cat, brand, 0.0, "", {}))
            continue
        out.append(Product(pid, name, cat, brand, float(g.price), g.description, dict(g.specs)))
    _log.info("enrich.done", count=len(out))
    return out


async def _invent_category(llm, category: str, n: int, existing_names: list[str]) -> list[Product]:
    avoid = "、".join(existing_names) if existing_names else "（无）"
    prompt = [
        SystemMessage(content=_INVENT_SYS),
        HumanMessage(
            content=(
                f"<category>{category}</category>\n"
                f"<count>{n}</count>\n"
                f"<avoid_names>{avoid}</avoid_names>"
            )
        ),
    ]
    res = await llm.chat("main_primary", prompt, structured=_ProductBatch)
    parsed: _ProductBatch = res.parsed
    out: list[Product] = []
    for g in parsed.products:
        out.append(
            Product(
                product_id="",  # assigned later
                product_name=g.product_name,
                category=category,
                brand=g.brand,
                price=float(g.price),
                description=g.description,
                specs=dict(g.specs),
            )
        )
    return out


async def _gen_faq(llm, topics: list[str]) -> list[FaqEntry]:
    payload = "、".join(topics)
    prompt = [
        SystemMessage(content=_FAQ_SYS),
        HumanMessage(content=f"<topics>{payload}</topics>\n<count>{len(topics)}</count>"),
    ]
    res = await llm.chat("main_primary", prompt, structured=_FaqBatch)
    parsed: _FaqBatch = res.parsed
    return [FaqEntry("", g.topic, g.question, g.answer) for g in parsed.faqs]


async def main() -> None:
    configure_logging(level="INFO", json=False)
    llm = get_llm_caller()

    # Pass 1: enrich seeds.
    seed_products = await _enrich_seed(llm)
    by_cat: dict[str, list[Product]] = {c: [] for c in CATEGORIES}
    for p in seed_products:
        by_cat.setdefault(p.category, []).append(p)

    # Pass 2: invent the remainder per category, in bounded-concurrency batches.
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _fill(cat: str) -> list[Product]:
        need = PER_CATEGORY - len(by_cat.get(cat, []))
        invented: list[Product] = []
        existing = [p.product_name for p in by_cat.get(cat, [])]
        while need > 0:
            take = min(NEW_PER_CALL, need)
            async with sem:
                batch = await _invent_category(
                    llm, cat, take, existing + [p.product_name for p in invented]
                )
            invented.extend(batch)
            need -= len(batch) if batch else take  # guard against empty batches
            if not batch:
                break
        return invented[: PER_CATEGORY - len(by_cat.get(cat, []))]

    fill_results = await asyncio.gather(*[_fill(c) for c in CATEGORIES])

    # Assemble + assign ids continuing from the seed max.
    products: list[Product] = list(seed_products)
    next_idx = len(SEED) + 1
    for _cat, invented in zip(CATEGORIES, fill_results, strict=True):
        for p in invented:
            p.product_id = f"P{next_idx:03d}"
            next_idx += 1
            products.append(p)

    # FAQ: topics in chunks, concurrent.
    faq_chunks = [FAQ_TOPICS[i : i + 5] for i in range(0, len(FAQ_TOPICS), 5)]
    # Repeat topics if needed to reach FAQ_TARGET (different phrasings).
    faq_results = await asyncio.gather(*[_gen_faq(llm, c) for c in faq_chunks])
    faqs: list[FaqEntry] = [f for chunk in faq_results for f in chunk]
    while len(faqs) < FAQ_TARGET:
        extra = await _gen_faq(llm, FAQ_TOPICS[: FAQ_TARGET - len(faqs)])
        if not extra:
            break
        faqs.extend(extra)
    faqs = faqs[:FAQ_TARGET]
    for i, f in enumerate(faqs, start=1):
        f.faq_id = f"FAQ{i:03d}"

    dump_jsonl(PRODUCTS_PATH, [p.to_dict() for p in products])
    dump_jsonl(FAQ_PATH, [f.to_dict() for f in faqs])
    _log.info("catalog.complete", products=len(products), faqs=len(faqs))
    print(f"wrote {len(products)} products -> {PRODUCTS_PATH}")
    print(f"wrote {len(faqs)} faqs -> {FAQ_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
