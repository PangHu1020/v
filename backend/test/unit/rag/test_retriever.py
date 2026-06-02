"""Unit tests for :class:`backend.v.rag.retriever.KnowledgeRetriever`."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from backend.v.rag.retriever import MAX_TOP_K, KnowledgeRetriever


def _fake_pool(rows: list[dict]) -> MagicMock:
    pool = MagicMock()
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    pool._conn = conn
    return pool


def _fake_embedder(vector: list[float] | None = None) -> MagicMock:
    e = MagicMock()
    e.aembed_query = AsyncMock(return_value=vector or [0.1] * 1024)
    return e


def _row(
    source_id: str,
    text: str,
    source_type: str = "product",
    similarity: float = 0.9,
) -> dict:
    return {
        "source_type": source_type,
        "source_id": source_id,
        "text": text,
        "metadata": {},
        "similarity": similarity,
    }


_r = KnowledgeRetriever()


class TestRetrieve:
    async def test_returns_rows(self) -> None:
        rows = [_row("P001", "iPhone"), _row("P002", "华为")]
        pool = _fake_pool(rows)
        result = await _r.retrieve(pool=pool, embedder=_fake_embedder(), query="手机")
        assert len(result) == 2
        assert result[0]["source_id"] == "P001"

    async def test_empty_query_returns_empty(self) -> None:
        assert await _r.retrieve(pool=_fake_pool([]), embedder=_fake_embedder(), query="") == []

    async def test_source_type_filter_sql(self) -> None:
        pool = _fake_pool([_row("F-04", "退货", source_type="faq")])
        await _r.retrieve(pool=pool, embedder=_fake_embedder(), query="退货", source_type="faq")
        call = pool._conn.fetch.await_args
        assert "WHERE source_type" in call.args[0]
        assert call.args[2] == "faq"

    async def test_no_filter_unfiltered_sql(self) -> None:
        pool = _fake_pool([_row("P001", "iPhone")])
        await _r.retrieve(pool=pool, embedder=_fake_embedder(), query="手机")
        call = pool._conn.fetch.await_args
        assert len(call.args) == 3
        assert "WHERE" not in call.args[0]

    async def test_top_k_clamped_low(self) -> None:
        pool = _fake_pool([_row("P001", "x")])
        await _r.retrieve(pool=pool, embedder=_fake_embedder(), query="x", top_k=0)
        assert pool._conn.fetch.await_args.args[-1] == 1

    async def test_top_k_clamped_high(self) -> None:
        pool = _fake_pool([_row("P001", "x")])
        await _r.retrieve(pool=pool, embedder=_fake_embedder(), query="x", top_k=9999)
        assert pool._conn.fetch.await_args.args[-1] == MAX_TOP_K


class TestFormat:
    def test_non_empty(self) -> None:
        rows = [_row("P001", "iPhone")]
        out = _r.format(rows)
        assert "[product:P001]" in out
        assert out.startswith("检索结果")

    def test_empty(self) -> None:
        assert _r.format([]) == "（未找到相关条目）"
