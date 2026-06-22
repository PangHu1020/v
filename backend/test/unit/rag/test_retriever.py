"""Unit tests for the Milvus hybrid+rerank retriever."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Mock pymilvus before project imports so retriever.py loads cleanly.
_pm = MagicMock()
for _a in (
    "MilvusClient",
    "AnnSearchRequest",
    "WeightedRanker",
    "DataType",
    "Function",
    "FunctionType",
):
    setattr(_pm, _a, MagicMock())
sys.modules["pymilvus"] = _pm

from backend.v.configs.base import MilvusSettings, RAGSettings  # noqa: E402
from backend.v.rag.retriever import KnowledgeRetriever, _build_filter_expr  # noqa: E402


def _fake_embedder(vector: list[float] | None = None) -> MagicMock:
    e = MagicMock()
    e.aembed_query = AsyncMock(return_value=vector or [0.1] * 1024)
    return e


def _row(source_id: str, text: str, source_type: str = "product", similarity: float = 0.9) -> dict:
    return {
        "source_type": source_type,
        "source_id": source_id,
        "text": text,
        "metadata": {},
        "similarity": similarity,
    }


def _hit(source_id: str, text: str, source_type: str = "product", score: float = 0.9) -> dict:
    return {
        "id": 1,
        "distance": score,
        "entity": {
            "source_type": source_type,
            "source_id": source_id,
            "text": text,
            "metadata": {},
        },
    }


def _settings(*, rerank_enabled: bool = False) -> MagicMock:
    cfg = MagicMock()
    cfg.milvus = MilvusSettings(uri="http://localhost:19530", token="", collection_name="test")
    cfg.rag = RAGSettings(
        min_k=2,
        dense_weight=0.5,
        bm25_weight=0.5,
        rerank_enabled=rerank_enabled,
        rerank_mode="local",
        rerank_candidates=20,
    )
    return cfg


class TestRetrieveHybrid:
    @patch("backend.v.rag.retriever.MilvusClient")
    async def test_hybrid_search_no_rerank(self, mock_cls: MagicMock) -> None:
        client = MagicMock()
        mock_cls.return_value = client
        client.has_collection.return_value = True
        client.hybrid_search.return_value = [
            [_hit("P001", "iPhone", score=0.9), _hit("P002", "iPad", score=0.8)]
        ]

        result = await KnowledgeRetriever().retrieve(
            embedder=_fake_embedder(), query="apple", settings=_settings()
        )

        assert [r["source_id"] for r in result] == ["P001", "P002"]
        # Single hybrid pass — no dense-only stage, no cascade.
        client.hybrid_search.assert_called_once()
        client.search.assert_not_called()

    @patch("backend.v.rag.retriever.MilvusClient")
    async def test_truncates_to_top_k(self, mock_cls: MagicMock) -> None:
        client = MagicMock()
        mock_cls.return_value = client
        client.has_collection.return_value = True
        client.hybrid_search.return_value = [
            [_hit(f"P{i:03d}", f"item {i}", score=0.9 - i * 0.05) for i in range(5)]
        ]

        result = await KnowledgeRetriever().retrieve(
            embedder=_fake_embedder(), query="x", top_k=3, settings=_settings()
        )
        assert len(result) == 3

    @patch("backend.v.rag.retriever.AnnSearchRequest")
    @patch("backend.v.rag.retriever.MilvusClient")
    async def test_keywords_drive_bm25_query_drives_dense(
        self, mock_cls: MagicMock, mock_ann: MagicMock
    ) -> None:
        """Dense branch embeds ``query``; BM25 branch gets ``keywords`` verbatim."""
        client = MagicMock()
        mock_cls.return_value = client
        client.has_collection.return_value = True
        client.hybrid_search.return_value = [[_hit("P001", "x")]]

        await KnowledgeRetriever().retrieve(
            embedder=_fake_embedder([0.1] * 1024),
            query="便宜的入门手机",
            keywords="红米 双卡",
            settings=_settings(),
        )

        # Two AnnSearchRequest builds: [0]=dense (embedding), [1]=sparse (BM25).
        dense_kw = mock_ann.call_args_list[0].kwargs
        sparse_kw = mock_ann.call_args_list[1].kwargs
        assert dense_kw["anns_field"] == "embedding"
        assert dense_kw["data"] == [[0.1] * 1024]  # the embedded query vector
        assert sparse_kw["anns_field"] == "sparse_embedding"
        assert sparse_kw["data"] == ["红米 双卡"]  # keywords, NOT the query

    @patch("backend.v.rag.retriever.AnnSearchRequest")
    @patch("backend.v.rag.retriever.MilvusClient")
    async def test_bm25_falls_back_to_query_when_no_keywords(
        self, mock_cls: MagicMock, mock_ann: MagicMock
    ) -> None:
        """No keywords → BM25 reuses the query (prior single-input behaviour)."""
        client = MagicMock()
        mock_cls.return_value = client
        client.has_collection.return_value = True
        client.hybrid_search.return_value = [[_hit("P001", "x")]]

        await KnowledgeRetriever().retrieve(
            embedder=_fake_embedder(), query="红米手机", settings=_settings()
        )

        sparse_kw = mock_ann.call_args_list[1].kwargs
        assert sparse_kw["data"] == ["红米手机"]

    @patch("backend.v.rag.retriever.build_reranker")
    @patch("backend.v.rag.retriever.MilvusClient")
    async def test_rerank_reorders(self, mock_cls: MagicMock, mock_build: MagicMock) -> None:
        client = MagicMock()
        mock_cls.return_value = client
        client.has_collection.return_value = True
        client.hybrid_search.return_value = [
            [
                _hit("P001", "a", score=0.9),
                _hit("P002", "b", score=0.8),
                _hit("P003", "c", score=0.7),
            ]
        ]
        # Reranker flips the order: P003 first.
        reranker = MagicMock()
        reranker.rerank = AsyncMock(return_value=[_row("P003", "c"), _row("P001", "a")])
        mock_build.return_value = reranker

        trace: dict = {}
        result = await KnowledgeRetriever().retrieve(
            embedder=_fake_embedder(),
            query="x",
            top_k=2,
            settings=_settings(rerank_enabled=True),
            trace=trace,
        )
        assert [r["source_id"] for r in result] == ["P003", "P001"]
        assert trace["reranked"] is True
        reranker.rerank.assert_awaited_once()

    @patch("backend.v.rag.retriever.MilvusClient")
    async def test_setup_collection_when_not_exists(self, mock_cls: MagicMock) -> None:
        client = MagicMock()
        mock_cls.return_value = client
        client.has_collection.return_value = False
        client.hybrid_search.return_value = [[_hit("P001", "x"), _hit("P002", "y")]]

        await KnowledgeRetriever().retrieve(
            embedder=_fake_embedder(), query="apple", settings=_settings()
        )

        client.create_collection.assert_called_once()
        assert client.create_index.call_count == 1
        client.load_collection.assert_called_once()

    async def test_empty_query_returns_empty(self) -> None:
        assert await KnowledgeRetriever().retrieve(embedder=_fake_embedder(), query="") == []

    async def test_milvus_client_none_raises_error(self) -> None:
        import pytest

        from backend.v.rag import retriever

        with patch.object(retriever, "MilvusClient", new=None):
            with pytest.raises(RuntimeError, match="pymilvus is not installed"):
                await retriever.KnowledgeRetriever().retrieve(
                    embedder=_fake_embedder(), query="x", settings=_settings()
                )


class TestFormat:
    def test_non_empty(self) -> None:
        out = KnowledgeRetriever().format([_row("P001", "iPhone")])
        assert "[product:P001]" in out
        assert out.startswith("检索结果")

    def test_empty(self) -> None:
        assert KnowledgeRetriever().format([]) == "（未找到相关条目）"


class TestBuildFilterExpr:
    def test_none_when_no_constraint(self) -> None:
        assert _build_filter_expr() is None

    def test_source_type_only(self) -> None:
        assert _build_filter_expr(source_type="product") == 'source_type == "product"'

    def test_category_only(self) -> None:
        assert _build_filter_expr(category="手机数码") == 'metadata["category"] == "手机数码"'

    def test_price_band_only(self) -> None:
        expr = _build_filter_expr(price_min=399.0, price_max=1300.0)
        assert expr == 'metadata["price"] >= 399.0 and metadata["price"] <= 1300.0'

    def test_price_min_only(self) -> None:
        assert _build_filter_expr(price_min=100.0) == 'metadata["price"] >= 100.0'

    def test_all_clauses_anded(self) -> None:
        expr = _build_filter_expr(
            source_type="product", category="鞋靴", price_min=359.0, price_max=429.0
        )
        assert expr == (
            'source_type == "product" and metadata["category"] == "鞋靴" '
            'and metadata["price"] >= 359.0 and metadata["price"] <= 429.0'
        )

    def test_zero_price_min_is_kept(self) -> None:
        # 0.0 is a valid bound — must not be dropped as falsy.
        assert _build_filter_expr(price_min=0.0) == 'metadata["price"] >= 0.0'

    def test_category_quote_escaped(self) -> None:
        # A double-quote in the value must not break out of the literal.
        expr = _build_filter_expr(category='a"b')
        assert expr == 'metadata["category"] == "a\\"b"'
