"""Per-task LLM caller with retry / fallback.

Single source of truth for "call DeepSeek with timeout, fall back on
recognized failures." Nodes do NOT call ``ChatOpenAI`` directly; they go
through :class:`LLMCaller` so the routing/fallback policy lives in one
place and is uniformly observable.

Trigger conditions for fallback (per locked CLAUDE.md decisions):

- ``asyncio.TimeoutError`` (after 30 seconds by default).
- ``openai.RateLimitError`` (HTTP 429).
- ``openai.InternalServerError`` (HTTP 5xx).

Same-family fallback only in Phase-1: ``main_primary`` -> ``main_fallback``.
Cross-family fallback (e.g., DeepSeek -> Qwen chat) is deferred to Phase-2.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage
from openai import APITimeoutError, InternalServerError, RateLimitError

from backend.v.configs.base import LLMSettings
from backend.v.models.factory import get_chat_model
from backend.v.utils.logging import get_logger

LLMRole = Literal["main_primary", "main_fallback", "summary", "memory_extract"]

_log = get_logger("models.llm_caller")

# Errors that should trigger same-family fallback once.
_FALLBACK_ERRORS: tuple[type[Exception], ...] = (
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    asyncio.TimeoutError,
)


@dataclass(frozen=True, slots=True)
class LLMResult:
    """The output of a successful LLM call plus accounting metadata."""

    message: AIMessage
    model: str
    role: LLMRole
    fallback_used: bool
    latency_ms: int


class LLMCaller:
    """Routes per-role chat invocations through DeepSeek with one fallback retry."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._timeout = settings.timeout_seconds

    async def chat(
        self,
        role: LLMRole,
        messages: list[BaseMessage],
        *,
        structured: type[Any] | None = None,
        tools: list[Any] | None = None,
    ) -> LLMResult:
        """Send ``messages`` through the configured chat model for ``role``.

        Args:
            role: Routing key. Determines model selection.
            messages: Prompt history.
            structured: Optional Pydantic model class to bind via
                ``with_structured_output``. The returned ``LLMResult.message``
                will be an ``AIMessage`` whose ``content`` is the JSON of the
                parsed model.
            tools: Optional list of LangChain tools to bind for the call.

        Returns:
            :class:`LLMResult`.

        Raises:
            The last exception observed if both primary and fallback fail.
        """
        primary_model = self._resolve(role)
        fallback_model = self._fallback_model_for(role)

        try:
            return await self._invoke(
                role=role,
                model=primary_model,
                messages=messages,
                structured=structured,
                tools=tools,
                fallback_used=False,
            )
        except _FALLBACK_ERRORS as primary_err:
            if not fallback_model or fallback_model == primary_model:
                raise
            _log.warning(
                "llm.primary_failed_falling_back",
                role=role,
                primary=primary_model,
                fallback=fallback_model,
                error=type(primary_err).__name__,
            )
            try:
                return await self._invoke(
                    role=role,
                    model=fallback_model,
                    messages=messages,
                    structured=structured,
                    tools=tools,
                    fallback_used=True,
                )
            except Exception as fb_err:
                _log.error(
                    "llm.fallback_also_failed",
                    role=role,
                    primary=primary_model,
                    fallback=fallback_model,
                    primary_error=type(primary_err).__name__,
                    fallback_error=type(fb_err).__name__,
                )
                raise primary_err from fb_err

    def _resolve(self, role: LLMRole) -> str:
        if role == "main_primary":
            return self._settings.main_primary
        if role == "main_fallback":
            return self._settings.main_fallback
        return self._settings.main_fallback

    def _fallback_model_for(self, role: LLMRole) -> str | None:
        # Same-family only in Phase-1: main_primary's only fallback is
        # main_fallback. Other roles already use the cheaper model and have
        # no fallback configured.
        if role == "main_primary":
            return self._settings.main_fallback
        return None

    async def _invoke(
        self,
        *,
        role: LLMRole,
        model: str,
        messages: list[BaseMessage],
        structured: type[Any] | None,
        tools: list[Any] | None,
        fallback_used: bool,
    ) -> LLMResult:
        chat = get_chat_model(self._settings, role if not fallback_used else "main_fallback")
        # Override the resolved model in case ``role`` and ``model`` diverge
        # (e.g., during fallback we resolve to main_fallback but the role
        # parameter is still the original).
        chat = chat.bind(model=model) if hasattr(chat, "bind") else chat

        runnable: Any = chat
        if structured is not None:
            runnable = chat.with_structured_output(structured, include_raw=False)
        elif tools:
            runnable = chat.bind_tools(tools)

        started = time.perf_counter()
        result = await asyncio.wait_for(runnable.ainvoke(messages), timeout=self._timeout)
        latency_ms = int((time.perf_counter() - started) * 1000)

        ai_message = result if isinstance(result, AIMessage) else AIMessage(content=str(result))
        _log.info(
            "llm.invoked",
            role=role,
            model=model,
            fallback_used=fallback_used,
            latency_ms=latency_ms,
        )
        return LLMResult(
            message=ai_message,
            model=model,
            role=role,
            fallback_used=fallback_used,
            latency_ms=latency_ms,
        )
