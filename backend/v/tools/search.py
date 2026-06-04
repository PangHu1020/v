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
) -> str:
    """Semantic search over the knowledge corpus (products / FAQ / policies).

    Use when the customer asks about something concrete in the catalog or
    knowledge base. Results carry source ids (e.g. ``[product:P001]``) for
    follow-up.

    Args:
        query: Natural-language description in Chinese. Be concrete.
        top_k: How many items to retrieve (1-20, default 5).
        source_type: Optional filter, e.g. ``"product"``. Omit for full corpus.
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
        settings=settings,  # None → retriever falls back to get_settings()
    )
    _log.info("tools.search.recalled", count=len(rows), source_type=source_type or "*")
    return _retriever.format(rows)
