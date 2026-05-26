"""Unit tests for ``backend.v.models.llm_caller``.

The tests patch the ``ChatOpenAI`` instance built by the factory so that
``ainvoke`` raises the desired error or returns a canned message. This
keeps the tests hermetic — no network, no httpx — while exercising the
real LLMCaller dispatch / fallback logic.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from openai import APITimeoutError, InternalServerError, RateLimitError

from backend.v.configs.base import LLMSettings
from backend.v.models.llm_caller import LLMCaller, LLMResult


@pytest.fixture
def settings() -> LLMSettings:
    return LLMSettings(
        _env_file=None,  # type: ignore[call-arg]
        base_url_deepseek="https://proxy/deepseek/v1",
        api_key_deepseek="sk-x",
        main_primary="deepseek-chat-v4-pro",
        main_fallback="deepseek-chat-v4-flash",
        timeout_seconds=2,
    )


def _bound_chat(behavior_per_call: list) -> MagicMock:
    """Build a mock chat model whose successive ``ainvoke`` calls follow ``behavior_per_call``."""
    chat = MagicMock()

    async def _ainvoke(messages):
        action = behavior_per_call.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action

    chat.ainvoke = AsyncMock(side_effect=_ainvoke)
    chat.bind.return_value = chat
    chat.with_structured_output.return_value = chat
    chat.bind_tools.return_value = chat
    return chat


@pytest.fixture
def patch_factory() -> Iterator[list]:
    """Replace ``get_chat_model`` so the LLMCaller picks up our mock."""
    behaviors: list = []
    chat = _bound_chat(behaviors)
    with patch("backend.v.models.llm_caller.get_chat_model", return_value=chat):
        # Yield the per-call behavior list; the test pushes onto it before
        # invoking ``LLMCaller.chat``.
        yield behaviors
        # Sanity: ensure the test consumed all the behaviors it queued.
        assert behaviors == [], f"unused behaviors: {behaviors}"


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://proxy/deepseek/v1/chat/completions")
    response = httpx.Response(status, request=request, json={"error": "x"})
    return httpx.HTTPStatusError("err", request=request, response=response)


def _openai_internal_error() -> InternalServerError:
    return InternalServerError(message="500", response=_http_error(500).response, body=None)


def _openai_rate_limit_error() -> RateLimitError:
    return RateLimitError(message="429", response=_http_error(429).response, body=None)


def _openai_timeout_error() -> APITimeoutError:
    request = httpx.Request("POST", "https://proxy/deepseek/v1/chat/completions")
    return APITimeoutError(request=request)


class TestPrimarySuccess:
    async def test_returns_ai_message(self, settings: LLMSettings, patch_factory: list) -> None:
        patch_factory.append(AIMessage(content="hello back"))
        caller = LLMCaller(settings)
        result = await caller.chat("main_primary", [HumanMessage(content="hi")])
        assert isinstance(result, LLMResult)
        assert result.message.content == "hello back"
        assert result.fallback_used is False
        assert result.role == "main_primary"
        assert result.model == "deepseek-chat-v4-pro"
        assert result.latency_ms >= 0


class TestFallbackTriggers:
    async def test_timeout_falls_back(self, settings: LLMSettings, patch_factory: list) -> None:
        patch_factory.extend([_openai_timeout_error(), AIMessage(content="from flash")])
        caller = LLMCaller(settings)
        result = await caller.chat("main_primary", [HumanMessage(content="hi")])
        assert result.fallback_used is True
        assert result.model == "deepseek-chat-v4-flash"
        assert result.message.content == "from flash"

    async def test_rate_limit_falls_back(self, settings: LLMSettings, patch_factory: list) -> None:
        patch_factory.extend([_openai_rate_limit_error(), AIMessage(content="ok")])
        caller = LLMCaller(settings)
        result = await caller.chat("main_primary", [HumanMessage(content="hi")])
        assert result.fallback_used is True

    async def test_5xx_falls_back(self, settings: LLMSettings, patch_factory: list) -> None:
        patch_factory.extend([_openai_internal_error(), AIMessage(content="ok")])
        caller = LLMCaller(settings)
        result = await caller.chat("main_primary", [HumanMessage(content="hi")])
        assert result.fallback_used is True

    async def test_asyncio_timeout_falls_back(
        self, settings: LLMSettings, patch_factory: list
    ) -> None:
        async def slow(_msgs):
            await asyncio.sleep(10)
            return AIMessage(content="too late")

        # Replace the patched chat ainvoke with a slow coroutine that
        # asyncio.wait_for will cancel.

        slow_chat = MagicMock()
        slow_chat.ainvoke = slow
        slow_chat.bind.return_value = slow_chat
        slow_chat.with_structured_output.return_value = slow_chat
        slow_chat.bind_tools.return_value = slow_chat

        # Successive get_chat_model calls: first slow, then fast for fallback.
        fast_chat = MagicMock()
        fast_chat.ainvoke = AsyncMock(return_value=AIMessage(content="from flash"))
        fast_chat.bind.return_value = fast_chat
        fast_chat.with_structured_output.return_value = fast_chat
        fast_chat.bind_tools.return_value = fast_chat

        with patch(
            "backend.v.models.llm_caller.get_chat_model",
            side_effect=[slow_chat, fast_chat],
        ):
            small_settings = LLMSettings(
                _env_file=None,  # type: ignore[call-arg]
                main_primary="pro",
                main_fallback="flash",
                timeout_seconds=0,  # 0 forces immediate timeout
            )
            caller = LLMCaller(small_settings)
            # Need a tiny positive timeout for asyncio.wait_for; 0 means raise immediately.
            caller._timeout = 0.01  # type: ignore[attr-defined]
            result = await caller.chat("main_primary", [HumanMessage(content="hi")])
            assert result.fallback_used is True
            assert result.model == "flash"

        # Drain unused expected behavior so the patch_factory autouse cleanup is happy.
        patch_factory.clear()


class TestNoFallback:
    async def test_summary_role_has_no_fallback(
        self, settings: LLMSettings, patch_factory: list
    ) -> None:
        patch_factory.append(_openai_rate_limit_error())
        caller = LLMCaller(settings)
        with pytest.raises(RateLimitError):
            await caller.chat("summary", [HumanMessage(content="x")])

    async def test_both_failing_raises_primary_error(
        self,
        settings: LLMSettings,
        patch_factory: list,
    ) -> None:
        patch_factory.extend([_openai_internal_error(), _openai_internal_error()])
        caller = LLMCaller(settings)
        with pytest.raises(InternalServerError):
            await caller.chat("main_primary", [HumanMessage(content="x")])
