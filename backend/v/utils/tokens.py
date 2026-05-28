"""Token counting utilities (Phase-3 Group B).

Used by the mid-session compression trigger and elsewhere we need to
estimate prompt size. Walks a fallback ladder so a missing tokenizer
never breaks the caller:

1. **LangChain message metadata** — if a ``BaseMessage`` already carries
   ``usage_metadata`` (filled by the LLM provider), trust it.
2. **tiktoken model-specific tokenizer** — when a ``model`` name is
   provided and tiktoken knows it.
3. **tiktoken cl100k_base** — the default for OpenAI-family models;
   accurate for English, slightly under-counts for CJK but close enough
   for thresholding.
4. **Character heuristic** — last resort, ``len(text) // 2`` (CJK runs
   ~2-3 BPE tokens per character; ASCII runs ~0.25; the average roughly
   halves to characters / 2).

The first three layers may import lazily and cache the encoder per
model name; the heuristic is a constant-time formula.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from langchain_core.messages import BaseMessage

from backend.v.utils.logging import get_logger

_log = get_logger("utils.tokens")


@lru_cache(maxsize=16)
def _encoder_for(model: str | None):  # type: ignore[no-untyped-def]
    """Return a tiktoken encoder, or ``None`` if tiktoken is unavailable.

    Caches one encoder per (resolved) name. Misses on the model fall
    back to ``cl100k_base``; failures (no tiktoken installed) return
    ``None`` and the caller drops to the character heuristic.
    """
    try:
        import tiktoken  # lazy import keeps the module light
    except ImportError:
        return None
    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            _log.debug("utils.tokens.unknown_model", model=model)
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        _log.warning("utils.tokens.tiktoken_failed", error=str(exc))
        return None


def count_text_tokens(text: str, model: str | None = None) -> int:
    """Return the token count for a string.

    See module docstring for the fallback ladder.
    """
    if not text:
        return 0
    encoder = _encoder_for(model)
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception as exc:
            _log.debug("utils.tokens.encode_failed", error=str(exc))
    # Character heuristic — see module docstring.
    return max(1, len(text) // 2)


def _message_metadata_tokens(message: BaseMessage) -> int | None:
    """Extract a token count from message metadata if the LLM provided it."""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        # AIMessage with usage_metadata: prefer the side that matches its role.
        # output_tokens for AI replies; input_tokens is for the prompt that
        # produced it (not us). For our "size of this message" use case,
        # output_tokens is the right pick on AIMessage; for non-AI messages
        # usage_metadata is rarely set, so we return None and fall through.
        out = usage.get("output_tokens")
        if isinstance(out, int) and out > 0:
            return out
    response_meta = getattr(message, "response_metadata", None)
    if isinstance(response_meta, dict):
        token_usage = response_meta.get("token_usage") or {}
        completion = token_usage.get("completion_tokens")
        if isinstance(completion, int) and completion > 0:
            return completion
    return None


def _content_to_text(content: Any) -> str:
    """Flatten a ``BaseMessage.content`` (which may be a list of parts) into a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content) if content is not None else ""


def count_message_tokens(message: BaseMessage, model: str | None = None) -> int:
    """Return the token count for one ``BaseMessage``.

    Prefers any token count the LLM provider already attached
    (``usage_metadata`` or ``response_metadata.token_usage``); otherwise
    falls back to encoding the textual content.
    """
    cached = _message_metadata_tokens(message)
    if cached is not None:
        return cached
    return count_text_tokens(_content_to_text(message.content), model)


def count_messages_tokens(messages: Iterable[BaseMessage], model: str | None = None) -> int:
    """Sum token counts across a list of messages."""
    return sum(count_message_tokens(m, model) for m in messages)


def reset_encoder_cache() -> None:
    """Drop the cached tokenizers. Useful in tests that monkeypatch tiktoken."""
    _encoder_for.cache_clear()
