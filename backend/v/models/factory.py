"""Build chat / embedding model instances from settings.

Both providers are addressed via OpenAI-compatible endpoints. Chat resolves a
``(base_url, api_key, model)`` triplet per role from :class:`LLMSettings`
(primary vs fallback endpoints); embedding uses its own independent
:class:`EmbeddingSettings`. Endpoint URLs/keys/model names come from ``.env``;
behaviour tunables (``temperature``, ``thinking``) from ``config.yaml``.

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

# Model families that accept the ``enable_thinking`` switch (DashScope Qwen3,
# SiliconFlow Nex, etc.). Matched case-insensitively against the model name.
# Everything else is treated as not-thinking-capable (the switch is dropped).
_THINKING_CAPABLE = re.compile(r"(qwen3|qwq|deepseek-r|deepseek-v3\.[1-9]|nex-)", re.IGNORECASE)

# Track which (model) we've already warned about, so a disabled-thinking
# request against an incapable model logs exactly once, not per call.
_warned_no_thinking: set[str] = set()


def _endpoint_for_role(settings: LLMSettings, role: ChatRole) -> tuple[str, str, str]:
    """Return (base_url, api_key, model) for a role.

    ``main_primary`` uses the primary endpoint; every other role (fallback,
    summary, memory_extract) uses the fallback endpoint — independent
    URL/key/model, so primary and fallback can be different providers.
    """
    if role == "main_primary":
        return settings.base_url, settings.api_key, settings.model
    return settings.base_url_fallback, settings.api_key_fallback, settings.model_fallback


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
    """Construct a ``ChatOpenAI`` for the given role's endpoint.

    Resolves the full (base_url, api_key, model) triplet from the role, so the
    returned client is already pointed at the right provider — no later
    ``.bind(model=...)`` override needed. ``settings.thinking`` is applied via
    ``extra_body`` only for models that support it (dropped + logged otherwise).
    """
    base_url, api_key, model = _endpoint_for_role(settings, role)
    extra_body = _thinking_extra_body(model, settings.thinking)
    kwargs: dict = {
        "model": model,
        "base_url": base_url or None,
        "api_key": SecretStr(api_key) if api_key else None,
        "temperature": settings.temperature,
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
    """Construct an ``OpenAIEmbeddings`` against the embedding endpoint.

    Embedding has its own URL/key/model (``EmbeddingSettings``), independent of
    chat — they're commonly different providers. ``settings`` (LLMSettings) is
    kept in the signature for call-site compatibility but no longer supplies
    embedding credentials.

    Two endpoint quirks handled here:

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
        base_url=embedding.base_url or None,
        api_key=SecretStr(embedding.api_key) if embedding.api_key else None,
        check_embedding_ctx_length=False,
    )
