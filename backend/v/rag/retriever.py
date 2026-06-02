"""Semantic retrieval over ``agent.knowledge_chunk`` via pgvector cosine.

Owns the SQL and formatting logic so multiple callers (the ``search``
tool, future admin endpoints, evaluation harness) share one
implementation.
"""

from __future__ import annotations

from typing import Any

from backend.v.utils.logging import get_logger

_log = get_logger("rag.retriever")

DEFAULT_TOP_K = 5
MAX_TOP_K = 20


def _format(rows: list[dict[str, Any]]) -> str:
    """Render retrieved rows as a compact Chinese list."""
    if not rows:
        return "（未找到相关条目）"
    lines = [f"- [{r['source_type']}:{r['source_id']}] {r['text']}" for r in rows]
    return "检索结果（按相关度排序）：\n" + "\n".join(lines)


class KnowledgeRetriever:
    """Stateless retriever; caller supplies pool and embedder per-call."""

    async def retrieve(
        self,
        *,
        pool: Any,
        embedder: Any,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        source_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run a cosine similarity query against ``agent.knowledge_chunk``.

        Args:
            pool: asyncpg connection pool.
            embedder: LangChain embeddings instance with ``aembed_query``.
            query: Natural-language query string.
            top_k: Maximum rows to return (clamped to [1, MAX_TOP_K]).
            source_type: Optional filter, e.g. ``"product"``.

        Returns:
            List of dicts with keys: source_type, source_id, text, metadata,
            similarity.
        """
        if not query:
            return []
        bounded_k = max(1, min(int(top_k), MAX_TOP_K))
        vector = await embedder.aembed_query(query)
        if source_type:
            sql = """
                SELECT source_type, source_id, text, metadata,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM agent.knowledge_chunk
                WHERE source_type = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
            """
            params: tuple[Any, ...] = (vector, source_type, bounded_k)
        else:
            sql = """
                SELECT source_type, source_id, text, metadata,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM agent.knowledge_chunk
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """
            params = (vector, bounded_k)
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

    def format(self, rows: list[dict[str, Any]]) -> str:
        """Render retrieved rows as a Chinese markdown list."""
        return _format(rows)
