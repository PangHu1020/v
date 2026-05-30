"""Unit tests for the ``search`` agent tool.

The pgvector / asyncpg surface is mocked: we wire a fake pool whose
``acquire().fetch()`` returns canned rows, plus an ``embedder`` whose
``aembed_query`` returns a fixed vector. The tests exercise:

- happy-path top-K rendering with ``[source_type:source_id]`` tags,
- ``source_type`` filter switches the SQL branch,
- empty corpus returns the placeholder string,
- missing pool / embedder degrades to the runtime-context error string,
- ``top_k`` is clamped to the [1, MAX_TOP_K] range.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from backend.v.tools.search import MAX_TOP_K, search


def _fake_pool(rows: list[dict]) -> MagicMock:
    pool = MagicMock()
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    pool._conn = conn  # exposed for assertions
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
    metadata: dict | None = None,
) -> dict:
    return {
        "source_type": source_type,
        "source_id": source_id,
        "text": text,
        "metadata": metadata or {},
        "similarity": similarity,
    }


class TestHappyPath:
    async def test_renders_top_k_with_source_tags(self) -> None:
        rows = [
            _row("P001", "iPhone 15 Pro 类别:手机数码 品牌:苹果 编号:P001"),
            _row("P002", "华为 Mate 60 类别:手机数码 品牌:华为 编号:P002", similarity=0.85),
        ]
        pool = _fake_pool(rows)
        embedder = _fake_embedder()
        config = {"configurable": {"pg_pool": pool, "embedder": embedder}}

        out = await search.ainvoke({"query": "苹果手机"}, config=config)

        assert "[product:P001]" in out
        assert "iPhone 15 Pro" in out
        assert "[product:P002]" in out
        assert out.startswith("检索结果")

    async def test_empty_corpus_returns_placeholder(self) -> None:
        pool = _fake_pool([])
        embedder = _fake_embedder()
        config = {"configurable": {"pg_pool": pool, "embedder": embedder}}

        out = await search.ainvoke({"query": "随便"}, config=config)

        assert out == "（未找到相关条目）"


class TestSourceTypeFilter:
    async def test_with_source_type_uses_filter_branch(self) -> None:
        pool = _fake_pool([_row("F-04", "退货政策...", source_type="faq")])
        embedder = _fake_embedder()
        config = {"configurable": {"pg_pool": pool, "embedder": embedder}}

        await search.ainvoke({"query": "怎么退货", "source_type": "faq"}, config=config)

        # fetch(sql, vector, source_type, top_k)
        call = pool._conn.fetch.await_args
        assert "WHERE source_type" in call.args[0]
        assert call.args[2] == "faq"
        assert call.args[3] >= 1

    async def test_without_source_type_uses_unfiltered_branch(self) -> None:
        pool = _fake_pool([_row("P001", "iPhone")])
        embedder = _fake_embedder()
        config = {"configurable": {"pg_pool": pool, "embedder": embedder}}

        await search.ainvoke({"query": "苹果手机"}, config=config)

        # fetch(sql, vector, top_k) — 3 positional args.
        call = pool._conn.fetch.await_args
        assert len(call.args) == 3
        assert "WHERE" not in call.args[0]


class TestRuntimeContext:
    async def test_missing_pool_returns_error(self) -> None:
        config = {"configurable": {"embedder": _fake_embedder()}}
        out = await search.ainvoke({"query": "x"}, config=config)
        assert out == "（无法检索：缺少运行上下文）"

    async def test_missing_embedder_returns_error(self) -> None:
        config = {"configurable": {"pg_pool": _fake_pool([])}}
        out = await search.ainvoke({"query": "x"}, config=config)
        assert out == "（无法检索：缺少运行上下文）"


class TestTopKClamp:
    async def test_top_k_below_one_clamped_to_one(self) -> None:
        pool = _fake_pool([_row("P001", "x")])
        embedder = _fake_embedder()
        config = {"configurable": {"pg_pool": pool, "embedder": embedder}}

        await search.ainvoke({"query": "x", "top_k": 0}, config=config)

        call = pool._conn.fetch.await_args
        assert call.args[-1] == 1

    async def test_top_k_above_max_clamped(self) -> None:
        pool = _fake_pool([_row("P001", "x")])
        embedder = _fake_embedder()
        config = {"configurable": {"pg_pool": pool, "embedder": embedder}}

        await search.ainvoke({"query": "x", "top_k": 9999}, config=config)

        call = pool._conn.fetch.await_args
        assert call.args[-1] == MAX_TOP_K
