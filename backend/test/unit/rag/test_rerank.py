"""Unit tests for ``backend.v.rag.rerank``."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx

from backend.v.rag.rerank import (
    LocalReranker,
    RemoteReranker,
    _rerank_by_scores,
    build_reranker,
)

_CANDS = [
    {"source_id": "P001", "text": "便宜手机 A"},
    {"source_id": "P002", "text": "贵的手机 B"},
    {"source_id": "P003", "text": "中端手机 C"},
]


def _mock_async_client(handler):
    """Return a context-manager mock whose .post calls ``handler(url, **kw)``."""

    class _Client:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def post(self, url: str, **kw: Any):
            return handler(url, **kw)

    return _Client


class _Resp:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)  # type: ignore[arg-type]

    def json(self) -> dict:
        return self._payload


class TestRerankByScores:
    def test_sorts_desc_and_truncates(self) -> None:
        out = _rerank_by_scores(_CANDS, [0.1, 0.9, 0.5], top_n=2)
        assert [r["source_id"] for r in out] == ["P002", "P003"]
        assert out[0]["rerank_score"] == 0.9

    def test_preserves_row_fields(self) -> None:
        out = _rerank_by_scores(_CANDS, [0.1, 0.2, 0.3], top_n=3)
        assert out[0]["source_id"] == "P003"
        assert out[0]["text"] == "中端手机 C"


class TestLocalReranker:
    async def test_reorders_by_sidecar_scores(self) -> None:
        def handler(url: str, **kw: Any):
            # sidecar returns results sorted by score, with passage text
            passages = kw["json"]["passages"]
            return _Resp(
                {
                    "results": [
                        {"score": 0.95, "passage": passages[2]},  # C
                        {"score": 0.80, "passage": passages[0]},  # A
                        {"score": 0.10, "passage": passages[1]},  # B
                    ]
                }
            )

        r = LocalReranker("http://localhost:8767/rerank")
        with patch("httpx.AsyncClient", _mock_async_client(handler)):
            out = await r.rerank("手机", _CANDS, top_n=2)
        assert [c["source_id"] for c in out] == ["P003", "P001"]

    async def test_empty_candidates(self) -> None:
        r = LocalReranker("http://x/rerank")
        assert await r.rerank("q", [], top_n=5) == []

    async def test_degrades_on_failure(self) -> None:
        def handler(url: str, **kw: Any):
            raise httpx.ConnectError("refused")

        r = LocalReranker("http://localhost:8767/rerank")
        with patch("httpx.AsyncClient", _mock_async_client(handler)):
            out = await r.rerank("手机", _CANDS, top_n=2)
        # degraded: original order, truncated
        assert [c["source_id"] for c in out] == ["P001", "P002"]


class TestRemoteReranker:
    async def test_reorders_by_indexed_scores(self) -> None:
        def handler(url: str, **kw: Any):
            assert kw["headers"]["Authorization"] == "Bearer sk-test"
            return _Resp(
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 2, "relevance_score": 0.4},
                    ]
                }
            )

        r = RemoteReranker("http://api/rerank", model="m", api_key="sk-test")
        with patch("httpx.AsyncClient", _mock_async_client(handler)):
            out = await r.rerank("手机", _CANDS, top_n=2)
        assert [c["source_id"] for c in out] == ["P002", "P003"]
        assert out[0]["rerank_score"] == 0.9

    async def test_degrades_on_garbled_response(self) -> None:
        def handler(url: str, **kw: Any):
            return _Resp({"unexpected": "shape"})

        r = RemoteReranker("http://api/rerank", model="m", api_key="k")
        with patch("httpx.AsyncClient", _mock_async_client(handler)):
            out = await r.rerank("手机", _CANDS, top_n=2)
        assert [c["source_id"] for c in out] == ["P001", "P002"]  # degraded

    async def test_degrades_on_http_error(self) -> None:
        def handler(url: str, **kw: Any):
            return _Resp({}, status=500)

        r = RemoteReranker("http://api/rerank", model="m", api_key="k")
        with patch("httpx.AsyncClient", _mock_async_client(handler)):
            out = await r.rerank("手机", _CANDS, top_n=3)
        assert [c["source_id"] for c in out] == ["P001", "P002", "P003"]


class TestBuildReranker:
    def test_local_default(self) -> None:
        assert isinstance(build_reranker(mode="local", url="http://x"), LocalReranker)

    def test_remote(self) -> None:
        r = build_reranker(mode="remote", url="http://x", model="m", api_key="k")
        assert isinstance(r, RemoteReranker)

    def test_unknown_mode_falls_back_to_local(self) -> None:
        assert isinstance(build_reranker(mode="weird", url="http://x"), LocalReranker)
