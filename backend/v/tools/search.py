"""``search``: generic semantic recall over ``agent.knowledge_chunk``.

This is the agent's catch-all retrieval tool for non-conversational
knowledge: products, FAQ, policy snippets, anything an operator (or a
seed script) has indexed into ``agent.knowledge_chunk``. The customer's
free-text query is embedded with Qwen ``text-embedding-v4`` (1024-dim)
and matched against the chunk corpus via pgvector's cosine operator.

Two knobs:

- ``top_k``: hard cap on returned rows.
- ``source_type``: optional filter, e.g. ``"product"`` to scope a query
  to the product catalog. Omit to search the entire corpus.

The result is a compact Chinese rendering — chunk text + the source id
in brackets — so the agent can cite "P001" / "FAQ-04" back to the
customer or hand it to a follow-up tool. The tool deliberately does NOT
do re-ranking by recency / importance the way ``recall_memory`` does:
knowledge corpora are roughly equally fresh, and the recency bias would
suppress evergreen answers.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from backend.v.utils.logging import get_logger

_log = get_logger("tools.search")

DEFAULT_TOP_K = 5
MAX_TOP_K = 20


def _format(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "（未找到相关条目）"
    lines = []
    for r in rows:
        tag = f"[{r['source_type']}:{r['source_id']}]"
        lines.append(f"- {tag} {r['text']}")
    return "检索结果（按相关度排序）：\n" + "\n".join(lines)


async def _search(
    *,
    pool: Any,
    embedder: Any,
    query: str,
    top_k: int,
    source_type: str | None,
) -> list[dict[str, Any]]:
    if not query:
        return []
    vector = await embedder.aembed_query(query)
    if source_type:
        sql = """
            SELECT
                source_type,
                source_id,
                text,
                metadata,
                1 - (embedding <=> $1::vector) AS similarity
            FROM agent.knowledge_chunk
            WHERE source_type = $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
        """
        params: tuple[Any, ...] = (vector, source_type, top_k)
    else:
        sql = """
            SELECT
                source_type,
                source_id,
                text,
                metadata,
                1 - (embedding <=> $1::vector) AS similarity
            FROM agent.knowledge_chunk
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """
        params = (vector, top_k)
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return [
        {
            "source_type": r["source_type"],
            "source_id": r["source_id"],
            "text": r["text"],
            "metadata": dict(r["metadata"] or {}),
            "similarity": float(r["similarity"]),
        }
        for r in rows
    ]


@tool("search", parse_docstring=True)
async def search(
    query: str,
    config: RunnableConfig,
    top_k: int = DEFAULT_TOP_K,
    source_type: str | None = None,
) -> str:
    """Semantic search over the knowledge corpus (products / FAQ / policies).

    Use this when the customer asks about something concrete that lives
    in the catalog or knowledge base: a product by description ("有没有
    保暖的羽绒服"), a FAQ topic ("怎么申请退货"), a policy detail
    ("会员有什么权益"). Returned items carry their source id (e.g.
    ``[product:P001]``) so a follow-up tool can drill in.

    Args:
        query: A natural-language description in Chinese. Be concrete.
        top_k: How many items to retrieve (1-20, default 5).
        source_type: Optional filter, e.g. ``"product"``. Omit to search
            everything.
    """
    cfg = config.get("configurable", {}) if config else {}
    pool = cfg.get("pg_pool")
    embedder = cfg.get("embedder")
    if not (pool and embedder):
        return "（无法检索：缺少运行上下文）"

    bounded_k = max(1, min(int(top_k), MAX_TOP_K))
    rows = await _search(
        pool=pool,
        embedder=embedder,
        query=query,
        top_k=bounded_k,
        source_type=source_type,
    )
    _log.info(
        "tools.search.recalled",
        count=len(rows),
        source_type=source_type or "*",
        query_len=len(query),
    )
    return _format(rows)
