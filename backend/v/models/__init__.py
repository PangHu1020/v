"""LLM model factory + retry/fallback caller.

Phase-1 ships:

- :class:`LLMCaller`: per-task routing + 30s timeout + same-family fallback
  (e.g., DeepSeek pro -> flash) on ``APITimeoutError`` / 5xx / 429.
- :func:`get_chat_model`: builds ``ChatOpenAI`` for chat completions against
  the OpenAI-compatible DeepSeek endpoint.
- :func:`get_embedding`: builds ``OpenAIEmbeddings`` against the OpenAI-
  compatible Qwen (DashScope) endpoint with the configured embedding dim.
"""

from backend.v.models.factory import get_chat_model, get_embedding
from backend.v.models.llm_caller import LLMCaller, LLMResult, LLMRole

__all__ = ["LLMCaller", "LLMResult", "LLMRole", "get_chat_model", "get_embedding"]
