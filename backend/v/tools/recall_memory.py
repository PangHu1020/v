"""``recall_memory``: long-term episodic recall via pgvector.

When the agent needs context that's not in the current session — past
preferences, prior incidents, repeated complaints — it calls this tool
with a natural-language query. We embed the query (Qwen
``text-embedding-v4`` @ 1024-dim), run a cosine similarity search
against ``agent.memory_episodes`` for that customer, and apply a
recency-weighted re-ranking before returning the top-K formatted
results.

Recency weighting: an episode's effective score is ::

    weighted = cosine_similarity * exp(-age_days / half_life_days)

with a 30-day half-life. Episodes from the same week as the query are
ranked roughly equal to "perfect match from a year ago, multiplied by
0.5", which approximates how a human relationship tracks recent
interactions over old ones.
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


def _format_episodes(rows: list[dict[str, Any]]) -> str:
    """Render the recall results as a compact Chinese paragraph for the agent."""
    if not rows:
        return "（暂无相关历史记忆）"
    lines = []
    for r in rows:
        age = r["age_days"]
        when = "今天" if age < 1 else f"{int(age)} 天前"
        lines.append(f"- ({when}) {r['content']}")
    return "客户历史记忆（按相关度+时效综合排序）：\n" + "\n".join(lines)


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
    """Embed the query and query pgvector with recency-weighted re-rank."""
    if not query:
        return []
    vector = await embedder.aembed_query(query)
    sql = """
        WITH ranked AS (
            SELECT
                id,
                content,
                metadata,
                created_at,
                EXTRACT(EPOCH FROM (now() - created_at)) / 86400 AS age_days,
                1 - (embedding <=> $1::vector) AS similarity
            FROM agent.memory_episodes
            WHERE channel = $2 AND channel_user_id = $3
            ORDER BY embedding <=> $1::vector
            LIMIT $4
        )
        SELECT
            id, content, metadata, created_at, age_days, similarity,
            similarity * exp(- age_days / $5) AS weighted_score
        FROM ranked
        ORDER BY weighted_score DESC
        LIMIT $6
    """
    # Two-stage: first take 3*top_k by raw cosine, then rerank by
    # recency-weighted score and trim to top_k. The DB does both in one
    # round-trip via the CTE.
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
    # Touch last_accessed_at so future maintenance jobs can prune
    # never-recalled episodes if needed.
    if rows:
        ids = [r["id"] for r in rows]
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE agent.memory_episodes SET last_accessed_at = now() "
                "WHERE id = ANY($1::uuid[])",
                ids,
            )
    return [
        {
            "id": str(r["id"]),
            "content": r["content"],
            "metadata": dict(r["metadata"] or {}),
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
    """Retrieve relevant long-term memories about THIS customer.

    Use this BEFORE asking the customer for information they may have
    given previously: preferences, complaints, past order patterns,
    delivery instructions, etc. The result is a short list of past
    facts ranked by combined semantic relevance + recency.

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
        # Tool was invoked outside a properly-configured graph turn.
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
        return _format_episodes(rows)
