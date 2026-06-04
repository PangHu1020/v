"""Milvus-based cascade retrieval over the knowledge corpus.

Uses a three-step cascade pipeline:
1. Fast-path: Dense vector search only. If the top result score is high
   enough and returns enough results, return immediately.
2. Hybrid search: If the fast-path fails, perform a dense + BM25 hybrid search.
3. LLM rewrite: If hybrid search fails to find good results, use the LLM to
   rewrite the query structurally and perform hybrid search again.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from backend.v.configs.base import get_settings
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

QUERY_REWRITE_SYSTEM_PROMPT = """<role>
你是一个专业的检索查询重写助手。你的任务是将用户的输入（可能比较模糊、口语化，包含情绪或无用词）重写为适合向量检索（Embedding）和文本检索（BM25）的结构化检索请求。
</role>

<task>
分析用户的输入，提取出其核心的查询意图：
1. 重写查询（rewritten_query）：构造一个表意清晰、去除了口语和情绪化词汇、补充了可能缺失的上下文的中文检索句。
2. 关键词列表（keywords）：提取出 3-5 个最重要的实体词、商品名、品牌名或故障关键词，用于精确匹配检索（BM25）。
3. 渠道源过滤器（source_type_filter）：推断用户想找的内容类型。如果明确涉及产品细节、价格、库存等，推断为 "product"；如果涉及退换货政策、会员权益、邮费等，推断为 "faq"；如果不确定，留空。
</task>

<rules>
- 确保关键词是从用户输入和重写句中提炼的实词，不包含标点符号和停用词。
- 过滤掉情绪性语言（例如："你们怎么搞的"、"催一下"）。
- 输出必须符合提供的 JSON 格式。
</rules>"""


class QueryRewriteResult(BaseModel):
    """Structured output for LLM query rewrite."""

    rewritten_query: str = Field(description="The optimized query sentence.")
    keywords: list[str] = Field(description="Extracted keywords for text search (BM25).")
    source_type_filter: str | None = Field(
        default=None, description="Inferred source type filter (product, faq, etc.) or None."
    )


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

    async def _rewrite_query(self, llm_caller: Any, query: str) -> QueryRewriteResult:
        """Asynchronously call the LLM to rewrite the query."""
        if not llm_caller:
            _log.warning("rag.retriever.rewrite_skipped: no llm_caller")
            return QueryRewriteResult(
                rewritten_query=query, keywords=[query], source_type_filter=None
            )

        prompt = [
            SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=f"<original_query>\n{query}\n</original_query>"),
        ]

        try:
            # Use plain text call to avoid with_structured_output raising on fenced output.
            # deepseek-v3.1 (fallback) sometimes wraps JSON in ```json ... ``` fences.
            import json as _json
            import re as _re

            res = await llm_caller.chat(role="main_fallback", messages=prompt)
            raw = (res.message.content or "").strip()
            raw = _re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = _re.sub(r"\n?```$", "", raw.strip()).strip()
            data = _json.loads(raw)
            result = QueryRewriteResult.model_validate(data)
            _log.info("rag.retriever.rewritten", rewritten=result.model_dump(mode="json"))
            return result
        except Exception as exc:
            _log.error("rag.retriever.rewrite_failed", error=type(exc).__name__)

        return QueryRewriteResult(rewritten_query=query, keywords=[query], source_type_filter=None)

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
        """Synchronously perform the Milvus query operations."""
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
        llm_caller: Any = None,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        source_type: str | None = None,
        settings: Any = None,
        trace: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run the cascade retrieval pipeline against Milvus database.

        Args:
            pool: asyncpg connection pool (unused, for backward compatibility).
            embedder: LangChain embeddings instance with ``aembed_query``.
            llm_caller: LLMCaller instance for query rewrite.
            query: Natural-language query string.
            top_k: Maximum rows to return (clamped to [1, MAX_TOP_K]).
            source_type: Optional filter, e.g. ``"product"``.
            settings: Optional AppSettings override for testing.
            trace: Optional dict; when provided it is populated in-place with
                ``stage`` (1/2/3 = the cascade step that produced the returned
                results) and ``max_score``. Used by the offline eval harness
                for per-stage attribution; ignored in production.

        Returns:
            List of dicts with keys: source_type, source_id, text, metadata, similarity.
        """
        if not query:
            return []

        bounded_k = max(1, min(int(top_k), MAX_TOP_K))

        # 1. Resolve configurations
        cfg = settings or get_settings()
        milvus_cfg = cfg.milvus
        rag_cfg = cfg.rag

        # Initialize Milvus client and check/create collection schema
        client = self._get_client(milvus_cfg.uri, milvus_cfg.token)
        await asyncio.to_thread(self._setup_collection_sync, client, milvus_cfg.collection_name)

        # Generate dense embedding vector for original query
        query_vector = await embedder.aembed_query(query)

        # --- Stage 1: Dense-only retrieval ---
        _log.debug("rag.retriever.cascade.stage1", query=query)
        results = await asyncio.to_thread(
            self._execute_search_sync,
            client,
            milvus_cfg.collection_name,
            1,
            query,
            query_vector,
            bounded_k,
            source_type,
            rag_cfg.dense_weight,
            rag_cfg.bm25_weight,
        )

        max_score = max((r["similarity"] for r in results), default=0.0)
        if len(results) >= rag_cfg.min_k and max_score >= rag_cfg.min_score:
            _log.info(
                "rag.retriever.cascade.stage1_success",
                count=len(results),
                max_score=max_score,
            )
            if trace is not None:
                trace["stage"] = 1
                trace["max_score"] = max_score
            return results

        # --- Stage 2: Hybrid (dense + BM25) retrieval on original query ---
        _log.debug("rag.retriever.cascade.stage2", query=query, prev_max_score=max_score)
        results = await asyncio.to_thread(
            self._execute_search_sync,
            client,
            milvus_cfg.collection_name,
            2,
            query,
            query_vector,
            bounded_k,
            source_type,
            rag_cfg.dense_weight,
            rag_cfg.bm25_weight,
        )

        max_score = max((r["similarity"] for r in results), default=0.0)
        if len(results) >= rag_cfg.min_k and max_score >= rag_cfg.stage2_min_score:
            _log.info(
                "rag.retriever.cascade.stage2_success",
                count=len(results),
                max_score=max_score,
            )
            if trace is not None:
                trace["stage"] = 2
                trace["max_score"] = max_score
            return results

        # --- Stage 3: LLM rewrite + Hybrid retrieval ---
        _log.debug("rag.retriever.cascade.stage3", query=query, prev_max_score=max_score)
        rewrite_result = await self._rewrite_query(llm_caller, query)

        # Embed the rewritten query
        rewritten_vector = await embedder.aembed_query(rewrite_result.rewritten_query)

        # Merge the rewritten search filter if detected by LLM
        final_source_type = source_type or rewrite_result.source_type_filter

        # Use the keywords compiled by LLM for the BM25 query text
        keyword_query_text = " ".join(rewrite_result.keywords)

        results = await asyncio.to_thread(
            self._execute_search_sync,
            client,
            milvus_cfg.collection_name,
            3,
            keyword_query_text,
            rewritten_vector,
            bounded_k,
            final_source_type,
            rag_cfg.dense_weight,
            rag_cfg.bm25_weight,
        )

        max_score = max((r["similarity"] for r in results), default=0.0)
        _log.info(
            "rag.retriever.cascade.stage3_complete",
            count=len(results),
            max_score=max_score,
        )
        if trace is not None:
            trace["stage"] = 3
            trace["max_score"] = max_score
        return results

    def format(self, rows: list[dict[str, Any]]) -> str:
        """Render retrieved rows as a Chinese markdown list."""
        return _format(rows)
