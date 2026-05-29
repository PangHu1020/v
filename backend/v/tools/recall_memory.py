"""``recall_memory``: long-term episodic recall via pgvector on ``agent.event_memory``.

When the active session's ``<recent_events>`` injection truncates an
older fact the customer is now referring to, the agent can call this
tool with a natural-language query. We embed the query (Qwen
``text-embedding-v4`` @ 1024-dim), run a cosine similarity search
against ``agent.event_memory`` for that customer, and apply an
importance + recency-weighted re-rank before returning the top-K
formatted results.

Effective score per row::

    weighted = cosine_similarity
             * exp(-age_days / half_life_days)
             * (0.5 + 0.5 * importance)

The importance multiplier ranges 0.5 (importance=0) → 1.0
(importance=1) so an entry never gets fully zeroed out by low
importance, but a critical entry (importance=1) outranks a trivial one
of equal cosine + age.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from backend.v.utils.logging import bind_request, get_logger

_log = get_logger("tools.recall_memory")

DEFAULT_TOP_K = 5
DEFAULT_HALF_LIFE_DAYS = 30
MAX_TOP_K = 20


def _format_recall(rows: list[dict[str, Any]]) -> str:
    """Render the recall results as a compact Chinese paragraph for the agent."""
    if not rows:
        return "（暂无相关历史记忆）"
    lines = []
    for r in rows:
        age = r["age_days"]
        when = "今天" if age < 1 else f"{int(age)} 天前"
        kind_tag = f"[{r['kind']}]" if r.get("kind") else ""
        lines.append(f"- ({when}){kind_tag} {r['content']}")
    return "客户历史记忆（按相关度+时效+重要性���合排序）：\n" + "\n".join(lines)


async def _recall(
    *,
    pool: Any,
    embedder: Any,
    channel: str,
    channel_user_id: str,
    query: str,
    top_k: int,
    half_life_days: float,
) -> list[dict[str, Any]]:
    """Embed the query and query pgvector with importance + recency-weighted re-rank."""
    if not query:
        return []
    vector = await embedder.aembed_query(query)
    sql = """
        WITH ranked AS (
            SELECT
                id,
                content,
                kind,
                importance,
                keywords,
                created_at,
                EXTRACT(EPOCH FROM (now() - created_at)) / 86400 AS age_days,
                1 - (embedding <=> $1::vector) AS similarity
            FROM agent.event_memory
            WHERE channel = $2
              AND channel_user_id = $3
              AND (expires_at IS NULL OR expires_at > now())
            ORDER BY embedding <=> $1::vector
            LIMIT $4
        )
        SELECT
            id, content, kind, importance, keywords, created_at,
            age_days, similarity,
            similarity
              * exp(- age_days / $5)
              * (0.5 + 0.5 * importance) AS weighted_score
        FROM ranked
        ORDER BY weighted_score DESC
        LIMIT $6
    """
    candidate_pool = max(top_k * 3, top_k)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            sql,
            vector,
            channel,
            channel_user_id,
            candidate_pool,
            half_life_days,
            top_k,
        )

    if rows:
        ids = [r["id"] for r in rows]
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE agent.event_memory SET last_accessed_at = now() WHERE id = ANY($1::uuid[])",
                ids,
            )

    return [
        {
            "id": str(r["id"]),
            "content": r["content"],
            "kind": r["kind"],
            "importance": float(r["importance"]),
            "keywords": list(r["keywords"] or []),
            "age_days": float(r["age_days"]),
            "similarity": float(r["similarity"]),
            "weighted_score": float(r["weighted_score"]),
        }
        for r in rows
    ]


@tool("recall_memory")
async def recall_memory(
    query: str,
    config: RunnableConfig,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """Retrieve relevant historical memories about THIS customer.

    Use this BEFORE asking the customer for information they may have
    given previously: preferences, complaints, past order patterns,
    delivery instructions, etc. The injected ``<recent_events>`` block
    only carries the most-recent few; this tool fishes deeper.

    Args:
        query: A natural-language question or topic in Chinese.
        top_k: How many memories to retrieve (1-20, default 5).
    """
    cfg = config.get("configurable", {}) if config else {}
    pool = cfg.get("pg_pool")
    embedder = cfg.get("embedder")
    channel = cfg.get("channel")
    channel_user_id = cfg.get("channel_user_id")
    if not (pool and embedder and channel and channel_user_id):
        return "（无法访问历史记忆：缺少运行上下文）"

    bounded_k = max(1, min(int(top_k), MAX_TOP_K))
    with bind_request(channel=channel, channel_user_id=channel_user_id):
        rows = await _recall(
            pool=pool,
            embedder=embedder,
            channel=channel,
            channel_user_id=channel_user_id,
            query=query,
            top_k=bounded_k,
            half_life_days=cfg.get("recall_half_life_days", DEFAULT_HALF_LIFE_DAYS),
        )
        _log.info("tools.recall_memory.recalled", count=len(rows), query_len=len(query))
        return _format_recall(rows)
