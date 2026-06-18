"""Rerank layer for the retrieval pipeline.

After hybrid search returns a candidate set, a reranker re-scores each
(query, passage) pair with a cross-encoder and reorders by relevance — far
sharper than the fused dense+BM25 score, which never sees the two texts
together.

Two transports, one interface:

- :class:`LocalReranker` — POST to a self-hosted reranker sidecar (a
  cross-encoder model served over HTTP on the local network). Zero per-call
  cost.
- :class:`RemoteReranker` — POST to a hosted rerank API (bearer-authed).

Both are deliberately named by *transport* (local vs remote), not by the
model or vendor behind them, so swapping the underlying reranker doesn't
invalidate the name.

Failure is non-fatal by design: if the reranker is unreachable or errors, the
caller falls back to the original candidate order (see
:meth:`Reranker.rerank` contract) — a degraded ranking beats a crashed search.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import httpx

from backend.v.utils.logging import get_logger

_log = get_logger("rag.rerank")


@runtime_checkable
class Reranker(Protocol):
    """Re-score and reorder retrieval candidates by (query, passage) relevance."""

    async def rerank(
        self, query: str, candidates: list[dict[str, Any]], *, top_n: int
    ) -> list[dict[str, Any]]:
        """Return the top-``top_n`` candidates reordered by rerank score.

        Args:
            query: The search query the candidates should be scored against.
            candidates: Retrieval result rows; each must carry a ``text`` field
                (the passage to score). Other keys (``source_id``, ``metadata``,
                …) are preserved on the returned rows.
            top_n: Cap on returned rows after reordering.

        Returns:
            Up to ``top_n`` rows, highest rerank score first. On any reranker
            failure, returns the input order truncated to ``top_n`` (never
            raises) — a degraded order is better than a broken search.
        """
        ...


def _rerank_by_scores(
    candidates: list[dict[str, Any]], scores: list[float], *, top_n: int
) -> list[dict[str, Any]]:
    """Attach scores to candidates, sort desc, truncate. Pure + testable."""
    scored = list(zip(candidates, scores, strict=True))
    scored.sort(key=lambda cs: cs[1], reverse=True)
    out: list[dict[str, Any]] = []
    for row, score in scored[:top_n]:
        enriched = dict(row)
        enriched["rerank_score"] = float(score)
        out.append(enriched)
    return out


class LocalReranker:
    """Rerank via a self-hosted sidecar: POST ``{query, passages}`` → scores.

    The sidecar contract (matches the project's reranker service): request
    ``{"query": str, "passages": [str]}``, response
    ``{"results": [{"score": float, "passage": str}], ...}`` ordered by score.
    We map scored passages back onto the original candidate rows by index of
    the passage text we sent.
    """

    def __init__(self, url: str, *, timeout_seconds: float = 10.0) -> None:
        self._url = url
        self._timeout = timeout_seconds

    async def rerank(
        self, query: str, candidates: list[dict[str, Any]], *, top_n: int
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        passages = [c.get("text", "") for c in candidates]
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as http:
                resp = await http.post(self._url, json={"query": query, "passages": passages})
            resp.raise_for_status()
            results = resp.json().get("results", [])
            # Map each scored passage back to its candidate by passage text.
            # The sidecar returns results sorted; we re-derive per-candidate
            # scores by position so duplicate texts still align deterministically.
            score_by_passage: dict[str, float] = {}
            for item in results:
                p = item.get("passage", "")
                if p not in score_by_passage:
                    score_by_passage[p] = float(item.get("score", 0.0))
            scores = [score_by_passage.get(p, float("-inf")) for p in passages]
        except Exception as exc:
            _log.warning("rag.rerank.local_failed", error=type(exc).__name__, url=self._url)
            return candidates[:top_n]
        return _rerank_by_scores(candidates, scores, top_n=top_n)


class RemoteReranker:
    """Rerank via a hosted API: POST ``{model, query, documents}`` → indexed scores.

    Targets the common hosted-rerank shape (DashScope / Cohere-style): response
    ``{"results": [{"index": int, "relevance_score": float}], ...}``. Bearer
    auth via ``api_key``.
    """

    def __init__(
        self, url: str, *, model: str, api_key: str, timeout_seconds: float = 15.0
    ) -> None:
        self._url = url
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_seconds

    async def rerank(
        self, query: str, candidates: list[dict[str, Any]], *, top_n: int
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        documents = [c.get("text", "") for c in candidates]
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as http:
                resp = await http.post(
                    self._url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "model": self._model,
                        "query": query,
                        "documents": documents,
                        "top_n": top_n,
                    },
                )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            # Hosted APIs return [{index, relevance_score}] (possibly already
            # sorted + truncated to top_n). Reorder candidates by returned index.
            out: list[dict[str, Any]] = []
            for item in results[:top_n]:
                idx = item.get("index")
                if idx is None or not (0 <= idx < len(candidates)):
                    continue
                enriched = dict(candidates[idx])
                enriched["rerank_score"] = float(item.get("relevance_score", 0.0))
                out.append(enriched)
            if not out:  # unexpected empty/garbled response → degrade
                return candidates[:top_n]
            return out
        except Exception as exc:
            _log.warning("rag.rerank.remote_failed", error=type(exc).__name__, url=self._url)
            return candidates[:top_n]


def build_reranker(*, mode: str, url: str, model: str = "", api_key: str = "") -> Reranker:
    """Construct the reranker for the configured transport ``mode``.

    Args:
        mode: ``"local"`` (sidecar) or ``"remote"`` (hosted API).
        url: Endpoint URL.
        model: Model name (remote only).
        api_key: Bearer key (remote only).
    """
    if mode == "remote":
        return RemoteReranker(url, model=model, api_key=api_key)
    return LocalReranker(url)
