"""Unit tests for the ``search`` agent tool."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Must precede project imports so retriever.py sees a pymilvus stub.
sys.modules.setdefault("pymilvus", MagicMock())

from backend.v.rag.retriever import _format  # noqa: E402
from backend.v.tools.search import search  # noqa: E402


def _fake_embedder() -> MagicMock:
    e = MagicMock()
    e.aembed_query = AsyncMock(return_value=[0.1] * 1024)
    return e


def _row(source_id: str, text: str, source_type: str = "product", similarity: float = 0.9) -> dict:
    return {
        "source_type": source_type,
        "source_id": source_id,
        "text": text,
        "metadata": {},
        "similarity": similarity,
    }


class TestHappyPath:
    @patch("backend.v.tools.search._retriever")
    async def test_renders_top_k_with_source_tags(self, mock_retriever: MagicMock) -> None:
        rows = [_row("P001", "iPhone"), _row("P002", "华为", similarity=0.85)]
        mock_retriever.retrieve = AsyncMock(return_value=rows)
        mock_retriever.format.side_effect = _format
        config = {"configurable": {"embedder": _fake_embedder()}}

        out = await search.ainvoke({"query": "苹果手机"}, config=config)

        assert "[product:P001]" in out and out.startswith("检索结果")

    @patch("backend.v.tools.search._retriever")
    async def test_empty_corpus_returns_placeholder(self, mock_retriever: MagicMock) -> None:
        mock_retriever.retrieve = AsyncMock(return_value=[])
        mock_retriever.format.side_effect = _format
        config = {"configurable": {"embedder": _fake_embedder()}}

        assert await search.ainvoke({"query": "随便"}, config=config) == "（未找到相关条目）"


class TestSourceTypeFilter:
    @patch("backend.v.tools.search._retriever")
    async def test_source_type_forwarded(self, mock_retriever: MagicMock) -> None:
        mock_retriever.retrieve = AsyncMock(return_value=[])
        embedder = _fake_embedder()
        config = {"configurable": {"embedder": embedder}}

        await search.ainvoke({"query": "退货", "source_type": "faq"}, config=config)

        mock_retriever.retrieve.assert_called_once_with(
            pool=None, embedder=embedder, llm_caller=None, query="退货", top_k=5, source_type="faq"
        )


class TestRuntimeContext:
    async def test_missing_embedder_returns_error(self) -> None:
        out = await search.ainvoke({"query": "x"}, config={"configurable": {}})
        assert out == "（无法检索：缺少运行上下文）"
