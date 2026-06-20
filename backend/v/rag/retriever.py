"""Milvus-based retrieval over the knowledge corpus: hybrid search + rerank.

One hybrid pass (dense embedding + BM25 sparse, fused by ``WeightedRanker``)
produces a candidate pool, then a cross-encoder reranker re-scores and
truncates to ``top_k``. No cascade / confidence gate / LLM query-rewrite —
this is agentic RAG, so the query is already written by the agent LLM from
conversation context; a second LLM rewrite was redundant. Rerank does the
quality lifting instead. If reranking is disabled or the reranker is
unreachable, the hybrid order is returned as-is.
"""

from __future__ import annotations

import asyncio
from typing import Any

from backend.v.configs.base import get_settings
from backend.v.rag.rerank import build_reranker
from backend.v.utils.logging import get_logger

try:
    from pymilvus import (
        AnnSearchRequest,
        DataType,
        Function,
        FunctionType,
        MilvusClient,
        WeightedRanker,
    )
except ImportError:
    AnnSearchRequest = None
    DataType = None
    Function = None
    FunctionType = None
    MilvusClient = None
    WeightedRanker = None

_log = get_logger("rag.retriever")

DEFAULT_TOP_K = 5
MAX_TOP_K = 20


def _format(rows: list[dict[str, Any]]) -> str:
    """Render retrieved rows as a compact Chinese list."""
    if not rows:
        return "（未找到相关条目）"
    lines = [f"- [{r['source_type']}:{r['source_id']}] {r['text']}" for r in rows]
    return "检索结果（按相关度排序）：\n" + "\n".join(lines)


def _normalize_score(score: float) -> float:
    """Clamp score to [0.0, 1.0] range."""
    return max(0.0, min(float(score), 1.0))


class KnowledgeRetriever:
    """Cascade retriever that executes queries against Milvus database."""

    def __init__(self) -> None:
        self._client: Any | None = None

    def _get_client(self, uri: str, token: str) -> Any:
        if MilvusClient is None:
            raise RuntimeError("pymilvus is not installed in the current environment")
        if self._client is None:
            self._client = MilvusClient(uri=uri, token=token)
        return self._client

    def _setup_collection_sync(self, client: Any, collection_name: str) -> None:
        """Initialize Milvus collection and indices if they do not exist."""
        if client.has_collection(collection_name):
            client.load_collection(collection_name)
            return

        from pymilvus import CollectionSchema, FieldSchema

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096, enable_analyzer=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        bm25_function = Function(
            name="text_bm25_emb",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_embedding"],
        )

        schema = CollectionSchema(
            fields=fields, description="Knowledge chunks collection", functions=[bm25_function]
        )

        client.create_collection(collection_name=collection_name, schema=schema)

        # Index configuration. pymilvus 3.x requires an IndexParams object
        # (built via prepare_index_params), not raw dicts.
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_name="dense_idx",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 64},
        )
        index_params.add_index(
            field_name="sparse_embedding",
            index_name="sparse_idx",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        client.create_index(collection_name, index_params=index_params)
        client.load_collection(collection_name)

    def _execute_search_sync(
        self,
        client: Any,
        collection_name: str,
        stage: int,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        source_type: str | None,
        dense_weight: float,
        bm25_weight: float,
    ) -> list[dict[str, Any]]:
        """Synchronously perform the Milvus query operations.

        ``stage`` selects the search mode: ``1`` = dense-only, ``2`` = hybrid
        (dense + BM25). The cascade is gone, but the two modes are kept so the
        weights can still be ablated; production calls hybrid.
        """
        filter_expr = f"source_type == '{source_type}'" if source_type else None

        if stage == 1:
            # Step 1: Pure Dense Search
            res = client.search(
                collection_name=collection_name,
                data=[query_vector],
                anns_field="embedding",
                search_params={"metric_type": "COSINE"},
                limit=top_k,
                filter=filter_expr,
                output_fields=["source_type", "source_id", "text", "metadata"],
            )
            hits = res[0] if res else []
        else:
            # Step 2 & 3: BM25 + Dense Hybrid Search
            req_dense = AnnSearchRequest(
                data=[query_vector],
                anns_field="embedding",
                param={"metric_type": "COSINE"},
                limit=top_k,
                expr=filter_expr,
            )
            req_sparse = AnnSearchRequest(
                data=[query_text],
                anns_field="sparse_embedding",
                param={"metric_type": "BM25"},
                limit=top_k,
                expr=filter_expr,
            )
            res = client.hybrid_search(
                collection_name=collection_name,
                reqs=[req_dense, req_sparse],
                ranker=WeightedRanker(dense_weight, bm25_weight),
                limit=top_k,
                output_fields=["source_type", "source_id", "text", "metadata"],
            )
            hits = res[0] if res else []

        results = []
        for hit in hits:
            # Support both object attributes (production) and dict keys (mock / testing)
            entity = hit.get("entity", {}) if isinstance(hit, dict) else getattr(hit, "entity", {})
            distance = (
                hit.get("distance", 0.0)
                if isinstance(hit, dict)
                else getattr(hit, "score", getattr(hit, "distance", 0.0))
            )
            results.append(
                {
                    "source_type": entity.get("source_type"),
                    "source_id": entity.get("source_id"),
                    "text": entity.get("text"),
                    "metadata": dict(entity.get("metadata") or {}),
                    "similarity": _normalize_score(distance),
                }
            )
        return results

    async def retrieve(
        self,
        *,
        pool: Any = None,  # Kept for signature compatibility
        embedder: Any,
        llm_caller: Any = None,  # Kept for signature compatibility (unused)
        query: str,
        top_k: int = DEFAULT_TOP_K,
        source_type: str | None = None,
        settings: Any = None,
        trace: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search (dense + BM25) then rerank, truncated to ``top_k``.

        Args:
            pool: Unused (signature compatibility).
            embedder: LangChain embeddings instance with ``aembed_query``.
            llm_caller: Unused (the LLM query-rewrite stage was removed).
            query: Natural-language query — already written by the agent LLM.
            top_k: Maximum rows to return (clamped to [1, MAX_TOP_K]).
            source_type: Optional filter, e.g. ``"product"``.
            settings: Optional AppSettings override for testing.
            trace: Optional dict populated in-place with ``reranked`` (bool) and
                ``candidates`` (pool size) for the offline eval harness.

        Returns:
            List of dicts with keys: source_type, source_id, text, metadata,
            similarity (and ``rerank_score`` when reranked).
        """
        if not query:
            return []

        bounded_k = max(1, min(int(top_k), MAX_TOP_K))
        cfg = settings or get_settings()
        milvus_cfg = cfg.milvus
        rag_cfg = cfg.rag

        client = self._get_client(milvus_cfg.uri, milvus_cfg.token)
        await asyncio.to_thread(self._setup_collection_sync, client, milvus_cfg.collection_name)

        query_vector = await embedder.aembed_query(query)

        # Single hybrid pass. When reranking, pull a wider candidate pool so the
        # reranker has something to reorder; otherwise just fetch top_k.
        candidate_k = (
            max(bounded_k, rag_cfg.rerank_candidates) if rag_cfg.rerank_enabled else bounded_k
        )
        candidate_k = min(candidate_k, MAX_TOP_K * 4)  # hard ceiling

        results = await asyncio.to_thread(
            self._execute_search_sync,
            client,
            milvus_cfg.collection_name,
            2,  # hybrid
            query,
            query_vector,
            candidate_k,
            source_type,
            rag_cfg.dense_weight,
            rag_cfg.bm25_weight,
        )

        reranked = False
        if rag_cfg.rerank_enabled and results:
            reranker = build_reranker(
                url=rag_cfg.rerank_url,
                model=rag_cfg.rerank_model,
                api_key=rag_cfg.rerank_api_key,
            )
            before = [r["source_id"] for r in results[:bounded_k]]
            results = await reranker.rerank(query, results, top_n=bounded_k)
            reranked = [r["source_id"] for r in results] != before
        else:
            results = results[:bounded_k]

        _log.info(
            "rag.retriever.done",
            count=len(results),
            candidates=candidate_k,
            reranked=reranked,
        )
        if trace is not None:
            trace["reranked"] = reranked
            trace["candidates"] = candidate_k
        return results

    def format(self, rows: list[dict[str, Any]]) -> str:
        """Render retrieved rows as a Chinese markdown list."""
        return _format(rows)
