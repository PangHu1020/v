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
    top_k: int = 5,
    source_type: str | None = None,
    category: str | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
) -> str:
    """Semantic search over the knowledge corpus (products / FAQ / policies).

    Use when the customer asks about something concrete in the catalog or
    knowledge base. Results carry source ids (e.g. ``[product:P001]``) for
    follow-up.

    When the customer states a concrete product category or budget, pass the
    optional ``category`` / ``price_min`` / ``price_max`` filters: they narrow
    the catalog to the matching sub-set *before* ranking, which sharply
    improves precision (e.g. a "1000元左右的手机" query should set
    ``category="手机数码"``, ``price_max=1300``). Leave them unset when the
    customer is vague — an over-tight filter can exclude good matches.

    Args:
        query: Natural-language description in Chinese. Be concrete.
        top_k: How many items to retrieve (1-20, default 5).
        source_type: Optional filter, e.g. ``"product"``. Omit for full corpus.
        category: Optional product category, e.g. ``"手机数码"`` / ``"家用电器"``
            / ``"鞋靴"`` / ``"服饰"`` / ``"食品饮料"`` / ``"休闲零食"``. Set only
            when the customer named a concrete category.
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
    )
    return _retriever.format(rows)
