"""Emotion detection (Phase-3 Group F).

The entry node can short-circuit to ``transfer_to_human`` **before**
calling the LLM when the customer's message exceeds an anger/frustration
threshold. This avoids burning an expensive LLM call on an already-upset
customer and reduces hallucination risk (angry text often triggers
off-topic LLM completions).

Two implementations share the :class:`EmotionDetector` Protocol:

- :class:`KeywordEmotionDetector` — lightweight keyword scorer that
  runs in-process. Default; no additional dependencies. Accuracy is
  limited but good enough for obvious cases ("骂人了", repeated
  exclamation marks, explicit transfer requests).
- A real BERT microservice satisfies the same Protocol via duck typing
  when deployed (see ``docs/`` for the integration guide). Connect it
  via :class:`HttpEmotionDetector`.

The threshold is configured via ``MEMORY_EMOTION_THRESHOLD`` (default
0.8 on a 0-1 scale). Set to 1.0 to disable.
"""

from __future__ import annotations

import re
from typing import Protocol

from backend.v.utils.logging import get_logger

_log = get_logger("emotion")

DEFAULT_THRESHOLD = 0.80


class EmotionDetector(Protocol):
    """Contract for any emotion detection backend.

    Callers only need to call :meth:`score`; the return value is a float
    in ``[0, 1]`` where 1 means maximum anger/frustration. The protocol
    is deliberately minimal so a local keyword scorer and a remote BERT
    service can swap without touching the graph.
    """

    async def score(self, text: str) -> float:
        """Return an anger/frustration score for ``text`` in [0, 1]."""
        ...


# ── Keyword scorer ────────────────────────────────────────────────────────────

# Chinese anger indicators — sampled broadly; weights are heuristic.
_HIGH_ANGER_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"操你|草你妈|傻逼|狗日的|你妈"), 0.9),
    (re.compile(r"骂|辱|投诉.*天|举报"), 0.85),
    (re.compile(r"[！!]{3,}"), 0.75),
    (re.compile(r"退款\s*[不没]"), 0.7),
    (re.compile(r"极差|非常差|很差|超级烂"), 0.65),
    (re.compile(r"坑|骗|欺诈|假货"), 0.60),
    (re.compile(r"不满\b|失望\b|气死"), 0.55),
    (re.compile(r"差评|投诉"), 0.50),
    (re.compile(r"赔偿|索赔"), 0.45),
    (re.compile(r"不行|不对|不好|有问题"), 0.30),
]

# Positive phrases that partially offset negative signals.
_CALM_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"谢谢|感谢|辛苦|帮帮忙"), -0.20),
    (re.compile(r"麻烦你|请问|你好"), -0.15),
]


class KeywordEmotionDetector:
    """Lightweight keyword scorer; no external dependencies.

    The algorithm: for each pattern that matches, take the **maximum**
    matching weight and add any calm offsets. Score is clamped to [0, 1].
    This deliberately avoids summing (which would fire on benign long
    messages) in favour of the single strongest anger signal minus
    courtesy markers.
    """

    async def score(self, text: str) -> float:
        if not text:
            return 0.0
        max_anger = 0.0
        for pattern, weight in _HIGH_ANGER_PATTERNS:
            if pattern.search(text):
                if weight > max_anger:
                    max_anger = weight
        if max_anger == 0.0:
            return 0.0
        offset = 0.0
        for pattern, delta in _CALM_PATTERNS:
            if pattern.search(text):
                offset += delta
        return max(0.0, min(1.0, max_anger + offset))


# ── HTTP stub ─────────────────────────────────────────────────────────────────


class HttpEmotionDetector:
    """Thin async HTTP client that calls a remote emotion-scoring endpoint.

    The endpoint must accept ``POST /score`` with body
    ``{"text": "..."}`` and return ``{"score": 0.0-1.0}``.
    Drop-in replacement for :class:`KeywordEmotionDetector`.
    """

    def __init__(self, base_url: str, timeout: float = 0.5) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def score(self, text: str) -> float:
        try:
            import httpx  # optional hot-path import

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/score",
                    json={"text": text},
                )
                resp.raise_for_status()
                return float(resp.json().get("score", 0.0))
        except Exception as exc:
            _log.warning("emotion.http.failed", error=type(exc).__name__)
            return 0.0


# ── Helper used by graph nodes ─────────────────────────────────────────────────


async def should_preempt_handoff(
    text: str,
    *,
    detector: EmotionDetector,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """Return ``True`` when ``text`` is angry enough to skip the LLM entirely."""
    s = await detector.score(text)
    if s >= threshold:
        _log.info("emotion.preempt", score=s, threshold=threshold)
        return True
    return False
