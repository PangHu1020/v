"""``search``: semantic recall over ``agent.knowledge_chunk``.

Thin tool wrapper — retrieval logic lives in :mod:`backend.v.rag.retriever`.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from backend.v.rag.retriever import KnowledgeRetriever
from backend.v.utils.logging import get_logger

_log = get_logger("tools.search")
_retriever = KnowledgeRetriever()


@tool("search", parse_docstring=True)
async def search(
    query: str,
    config: RunnableConfig,
    keywords: str | None = None,
    top_k: int = 5,
    source_type: str | None = None,
    category: str | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
) -> str:
    """Semantic + keyword search over the knowledge corpus (products / FAQ).

    Use when the customer asks about something concrete in the catalog. Results
    carry source ids (e.g. ``[product:P001]``) for follow-up.

    Retrieval fuses two channels — craft each input for its channel:

    - ``query`` (dense embedding): the customer's *intent*, rewritten as a clean
      product description. STRIP colloquial filler and emotion — "老板我预算紧
      张随便看看那种便宜点的手机别太贵啊" becomes "便宜的入门手机". Keep the
      semantic core (use, attributes, scenario); drop chit-chat, complaints,
      politeness, hesitation.
    - ``keywords`` (BM25 lexical): the concrete *entities the customer named* —
      brand, model, category word, spec terms — space-separated, no sentences.
      e.g. "红米 双卡 大电池 5G". Omit words the customer didn't actually say;
      BM25 matches surface terms, so invented synonyms hurt. Leave unset if the
      customer gave no concrete entity.

    Then the scalar filters narrow the catalog *before* ranking — set them
    whenever the customer states a category or budget (big precision win):

    - ``category``: exact category, one of 手机数码 / 家用电器 / 鞋靴 / 服饰 /
      食品饮料 / 休闲零食. Set only when the customer named a concrete category.
    - ``price_min`` / ``price_max``: budget bounds in RMB; map loose phrasing to
      numbers ("一千出头" → ``price_max=1300``, "三千左右" → 2500–3500).

    Multi-need / multi-constraint asks: emit ONE search call per distinct need
    (they run in parallel) — e.g. "给妈妈买礼盒 + 自己囤的零食" → two calls,
    each with its own query/keywords/filters.

    Args:
        query: Clean semantic description for embedding — colloquial filler and
            emotion stripped, just the product intent.
        keywords: Space-separated entity keywords the customer actually said
            (brand / category / spec) for BM25. Omit when there is none.
        top_k: How many items to retrieve (1-20, default 5).
        source_type: Optional filter, e.g. ``"product"``. Omit for full corpus.
        category: Optional exact product category (see list above).
        price_min: Optional inclusive lower price bound in RMB.
        price_max: Optional inclusive upper price bound in RMB.
    """
    cfg = config.get("configurable", {}) if config else {}
    embedder = cfg.get("embedder")
    llm_caller = cfg.get("llm_caller")
    settings = cfg.get("settings")  # AppSettings, carries milvus + rag config
    if not embedder:
        return "（无法检索：缺少运行上下文）"

    rows = await _retriever.retrieve(
        embedder=embedder,
        llm_caller=llm_caller,
        query=query,
        keywords=keywords,
        top_k=top_k,
        source_type=source_type,
        category=category,
        price_min=price_min,
        price_max=price_max,
        settings=settings,  # None → retriever falls back to get_settings()
    )
    _log.info(
        "tools.search.recalled",
        count=len(rows),
        source_type=source_type or "*",
        category=category or "*",
        price_min=price_min,
        price_max=price_max,
        has_keywords=bool(keywords and keywords.strip()),
    )
    return _retriever.format(rows)
