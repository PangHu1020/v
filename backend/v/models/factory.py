"""Build chat / embedding model instances from settings.

Both providers are addressed via OpenAI-compatible endpoints:

- DeepSeek: ``LLM_BASE_URL_DEEPSEEK`` / ``LLM_API_KEY_DEEPSEEK``
- Qwen (DashScope): ``LLM_BASE_URL_QWEN`` / ``LLM_API_KEY_QWEN``

The factory does not cache instances; the LLMCaller above caches per-role.
"""

from __future__ import annotations

import re
from typing import Literal

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from backend.v.configs.base import EmbeddingSettings, LLMSettings
from backend.v.utils.logging import get_logger

ChatRole = Literal["main_primary", "main_fallback", "summary", "memory_extract"]

_log = get_logger("models.factory")

# Model families that accept the DashScope ``enable_thinking`` switch.
# Matched case-insensitively against the model name. Everything else is
# treated as not-thinking-capable (the switch is silently dropped).
_THINKING_CAPABLE = re.compile(r"(qwen3|qwq|deepseek-r|deepseek-v3\.[1-9])", re.IGNORECASE)

# Track which (model) we've already warned about, so a disabled-thinking
# request against an incapable model logs exactly once, not per call.
_warned_no_thinking: set[str] = set()


def _model_for_role(settings: LLMSettings, role: ChatRole) -> str:
    if role == "main_primary":
        return settings.main_primary
    if role == "main_fallback":
        return settings.main_fallback
    # Phase-1: summary and memory_extract reuse the same flash-tier model
    # as the main fallback; Phase-2 may break these out into independent
    # routing keys.
    return settings.main_fallback


def _thinking_extra_body(model: str, thinking: bool) -> dict | None:
    """Resolve the ``extra_body`` for the thinking switch, with graceful degrade.

    - Thinking-capable model → always pass ``enable_thinking`` (True turns it
      on; False explicitly turns OFF the family default, e.g. Qwen3 which
      defaults thinking on and would otherwise add latency / reject streaming).
    - Incapable model + thinking requested → don't pass anything (no error, no
      forced thinking); log once that we fell back to no-thinking.
    - Incapable model + thinking off → nothing to do.
    """
    if _THINKING_CAPABLE.search(model):
        return {"enable_thinking": thinking}
    if thinking and model not in _warned_no_thinking:
        _warned_no_thinking.add(model)
        _log.info(
            "models.factory.thinking_unsupported",
            model=model,
            note="model does not support thinking; using no-thinking mode",
        )
    return None


def get_chat_model(settings: LLMSettings, role: ChatRole) -> ChatOpenAI:
    """Construct a ``ChatOpenAI`` for the given role against DeepSeek.

    The ``settings.thinking`` flag is applied via ``extra_body`` only for
    models that support it; for others it is silently dropped (logged once).
    """
    model = _model_for_role(settings, role)
    extra_body = _thinking_extra_body(model, settings.thinking)
    kwargs: dict = {
        "model": model,
        "base_url": settings.base_url_deepseek or None,
        "api_key": SecretStr(settings.api_key_deepseek) if settings.api_key_deepseek else None,
        "timeout": settings.timeout_seconds,
        "max_retries": 0,  # LLMCaller handles fallback; we want a single attempt here.
    }
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    return ChatOpenAI(**kwargs)


def get_embedding(
    settings: LLMSettings,
    embedding: EmbeddingSettings,
) -> OpenAIEmbeddings:
    """Construct an ``OpenAIEmbeddings`` against the Qwen (DashScope) endpoint.

    Two DashScope-specific quirks are handled here:

    - ``check_embedding_ctx_length=False``: langchain-openai otherwise
      tiktoken-encodes inputs into token-id lists before sending. DashScope's
      OpenAI-compatible endpoint rejects token-id input with a 400
      ("contents is neither str nor list of str"); it only accepts raw
      strings. Disabling the check sends the text through verbatim.
    - ``dimensions`` is omitted: ``text-embedding-v4`` returns 1024-dim
      vectors by default, matching ``EMBEDDING_DIM`` and the Milvus
      collection schema.
    """
    return OpenAIEmbeddings(
        model=embedding.model,
        base_url=settings.base_url_qwen or None,
        api_key=SecretStr(settings.api_key_qwen) if settings.api_key_qwen else None,
        check_embedding_ctx_length=False,
    )
