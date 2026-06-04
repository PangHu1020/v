"""Unit tests for the Milvus cascade retriever."""

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
from backend.v.rag.retriever import KnowledgeRetriever, QueryRewriteResult  # noqa: E402


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


def _settings() -> MagicMock:
    cfg = MagicMock()
    cfg.milvus = MilvusSettings(uri="http://localhost:19530", token="", collection_name="test")
    cfg.rag = RAGSettings(min_score=0.7, min_k=2, dense_weight=0.5, bm25_weight=0.5)
    return cfg


class TestRetrieveCascade:
    @patch("backend.v.rag.retriever.MilvusClient")
    async def test_stage1_success_fast_path(self, mock_cls: MagicMock) -> None:
        client = MagicMock()
        mock_cls.return_value = client
        client.has_collection.return_value = True
        client.search.return_value = [
            [_hit("P001", "iPhone", score=0.85), _hit("P002", "iPad", score=0.80)]
        ]

        result = await KnowledgeRetriever().retrieve(
            embedder=_fake_embedder(), query="apple", settings=_settings()
        )

        assert len(result) == 2
        assert result[0]["similarity"] == 0.85
        client.search.assert_called_once()
        client.hybrid_search.assert_not_called()

    @patch("backend.v.rag.retriever.MilvusClient")
    async def test_stage2_fallback_hybrid(self, mock_cls: MagicMock) -> None:
        client = MagicMock()
        mock_cls.return_value = client
        client.has_collection.return_value = True
        client.search.return_value = [[_hit("P001", "iPhone", score=0.5)]]
        client.hybrid_search.return_value = [
            [_hit("P001", "iPhone", score=0.9), _hit("P002", "iPad", score=0.8)]
        ]

        result = await KnowledgeRetriever().retrieve(
            embedder=_fake_embedder(), query="apple", settings=_settings()
        )

        assert len(result) == 2
        client.search.assert_called_once()
        client.hybrid_search.assert_called_once()

    @patch("backend.v.rag.retriever.MilvusClient")
    async def test_stage3_fallback_llm_rewrite(self, mock_cls: MagicMock) -> None:
        client = MagicMock()
        mock_cls.return_value = client
        client.has_collection.return_value = True
        client.search.return_value = [[_hit("P001", "x", score=0.5)]]
        client.hybrid_search.side_effect = [
            [[_hit("P001", "x", score=0.5)]],
            [[_hit("P002", "iPad Mini", score=0.85)]],
        ]
        llm = MagicMock()
        res = MagicMock()
        res.parsed = QueryRewriteResult(
            rewritten_query="苹果平板", keywords=["苹果", "平板"], source_type_filter=None
        )
        llm.chat = AsyncMock(return_value=res)

        result = await KnowledgeRetriever().retrieve(
            embedder=_fake_embedder(), llm_caller=llm, query="买个平板", settings=_settings()
        )

        assert result[0]["source_id"] == "P002"
        client.search.assert_called_once()
        assert client.hybrid_search.call_count == 2
        llm.chat.assert_called_once()

    @patch("backend.v.rag.retriever.MilvusClient")
    async def test_setup_collection_when_not_exists(self, mock_cls: MagicMock) -> None:
        client = MagicMock()
        mock_cls.return_value = client
        client.has_collection.return_value = False
        client.search.return_value = [[_hit("P001", "x", score=0.9), _hit("P002", "y", score=0.8)]]

        await KnowledgeRetriever().retrieve(
            embedder=_fake_embedder(), query="apple", settings=_settings()
        )

        client.create_collection.assert_called_once()
        assert client.create_index.call_count == 1
        client.load_collection.assert_called_once()

    @patch("backend.v.rag.retriever.MilvusClient")
    async def test_rewrite_no_llm_caller(self, mock_cls: MagicMock) -> None:
        client = MagicMock()
        mock_cls.return_value = client
        client.has_collection.return_value = True
        client.search.return_value = [[_hit("P001", "x", score=0.5)]]
        client.hybrid_search.side_effect = [
            [[_hit("P001", "x", score=0.5)]],
            [[_hit("P001", "x", score=0.5)]],
        ]

        result = await KnowledgeRetriever().retrieve(
            embedder=_fake_embedder(), llm_caller=None, query="apple", settings=_settings()
        )

        assert isinstance(result, list)
        assert client.hybrid_search.call_count == 2

    @patch("backend.v.rag.retriever.MilvusClient")
    async def test_rewrite_llm_exception_recovers(self, mock_cls: MagicMock) -> None:
        client = MagicMock()
        mock_cls.return_value = client
        client.has_collection.return_value = True
        client.search.return_value = [[_hit("P001", "x", score=0.5)]]
        client.hybrid_search.side_effect = [
            [[_hit("P001", "x", score=0.5)]],
            [[_hit("P001", "x", score=0.6)]],
        ]
        llm = MagicMock()
        llm.chat = AsyncMock(side_effect=RuntimeError("LLM offline"))

        result = await KnowledgeRetriever().retrieve(
            embedder=_fake_embedder(), llm_caller=llm, query="apple", settings=_settings()
        )

        assert isinstance(result, list)
        llm.chat.assert_called_once()

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
