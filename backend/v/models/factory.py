"""Build chat / embedding model instances from settings.

Both providers are addressed via OpenAI-compatible endpoints:

- DeepSeek: ``LLM_BASE_URL_DEEPSEEK`` / ``LLM_API_KEY_DEEPSEEK``
- Qwen (DashScope): ``LLM_BASE_URL_QWEN`` / ``LLM_API_KEY_QWEN``

The factory does not cache instances; the LLMCaller above caches per-role.
"""

from __future__ import annotations

from typing import Literal

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from backend.v.configs.base import EmbeddingSettings, LLMSettings

ChatRole = Literal["main_primary", "main_fallback", "summary", "memory_extract"]


def _model_for_role(settings: LLMSettings, role: ChatRole) -> str:
    if role == "main_primary":
        return settings.main_primary
    if role == "main_fallback":
        return settings.main_fallback
    # Phase-1: summary and memory_extract reuse the same flash-tier model
    # as the main fallback; Phase-2 may break these out into independent
    # routing keys.
    return settings.main_fallback


def get_chat_model(settings: LLMSettings, role: ChatRole) -> ChatOpenAI:
    """Construct a ``ChatOpenAI`` for the given role against DeepSeek."""
    return ChatOpenAI(
        model=_model_for_role(settings, role),
        base_url=settings.base_url_deepseek or None,
        api_key=SecretStr(settings.api_key_deepseek) if settings.api_key_deepseek else None,
        timeout=settings.timeout_seconds,
        max_retries=0,  # LLMCaller handles fallback; we want a single attempt here.
    )


def get_embedding(
    settings: LLMSettings,
    embedding: EmbeddingSettings,
) -> OpenAIEmbeddings:
    """Construct an ``OpenAIEmbeddings`` against the Qwen endpoint.

    Note: DashScope's text-embedding-v3 does not support the `dimensions`
    parameter via the OpenAI-compatible endpoint; it always returns 1024-dim
    vectors. The parameter is omitted to avoid 400 errors.
    """
    return OpenAIEmbeddings(
        model=embedding.model,
        base_url=settings.base_url_qwen or None,
        api_key=SecretStr(settings.api_key_qwen) if settings.api_key_qwen else None,
    )
