"""Unit tests for ``backend.v.models.factory``."""

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend.v.configs.base import EmbeddingSettings, LLMSettings
from backend.v.models.factory import get_chat_model, get_embedding


@pytest.fixture
def llm_settings() -> LLMSettings:
    return LLMSettings(
        _env_file=None,  # type: ignore[call-arg]
        base_url="https://api.siliconflow.cn/v1",
        api_key="sk-primary-test",
        model="deepseek-chat-v4-pro",
        base_url_fallback="https://proxy.example.com/fallback/v1",
        api_key_fallback="sk-fallback-test",
        model_fallback="deepseek-chat-v4-flash",
        timeout_seconds=30,
    )


@pytest.fixture
def embedding_settings() -> EmbeddingSettings:
    return EmbeddingSettings(
        _env_file=None,  # type: ignore[call-arg]
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-embed-test",
        model="text-embedding-v4",
        dim=1024,
    )


class TestGetChatModel:
    def test_main_primary_uses_pro(self, llm_settings: LLMSettings) -> None:
        chat = get_chat_model(llm_settings, "main_primary")
        assert isinstance(chat, ChatOpenAI)
        assert chat.model_name == "deepseek-chat-v4-pro"

    def test_main_fallback_uses_flash(self, llm_settings: LLMSettings) -> None:
        chat = get_chat_model(llm_settings, "main_fallback")
        assert chat.model_name == "deepseek-chat-v4-flash"

    def test_summary_and_extract_use_flash(self, llm_settings: LLMSettings) -> None:
        for role in ("summary", "memory_extract"):
            chat = get_chat_model(llm_settings, role)  # type: ignore[arg-type]
            assert chat.model_name == "deepseek-chat-v4-flash"

    def test_no_inner_retries(self, llm_settings: LLMSettings) -> None:
        chat = get_chat_model(llm_settings, "main_primary")
        # LLMCaller owns retry/fallback; the wrapped client must not retry.
        assert chat.max_retries == 0

    def test_timeout_applied(self, llm_settings: LLMSettings) -> None:
        chat = get_chat_model(llm_settings, "main_primary")
        assert chat.request_timeout == 30


class TestThinking:
    """The thinking switch: applied to capable models, gracefully dropped otherwise."""

    def test_capable_model_passes_enable_thinking_true(self) -> None:
        from backend.v.models.factory import _thinking_extra_body

        assert _thinking_extra_body("qwen3.6-max", thinking=True) == {"enable_thinking": True}
        assert _thinking_extra_body("qwq-32b", thinking=True) == {"enable_thinking": True}
        assert _thinking_extra_body("deepseek-r1", thinking=True) == {"enable_thinking": True}

    def test_capable_model_explicitly_disables(self) -> None:
        # thinking=False must still pass enable_thinking=False to turn OFF the
        # family default (Qwen3 defaults thinking ON).
        from backend.v.models.factory import _thinking_extra_body

        assert _thinking_extra_body("qwen3.6-plus", thinking=False) == {"enable_thinking": False}

    def test_incapable_model_thinking_on_degrades_silently(self) -> None:
        # No error, no forced thinking — just None (nothing passed).
        from backend.v.models import factory

        factory._warned_no_thinking.clear()
        assert factory._thinking_extra_body("deepseek-v4-flash", thinking=True) is None
        # Warned once, recorded so it won't spam.
        assert "deepseek-v4-flash" in factory._warned_no_thinking

    def test_incapable_model_thinking_off_is_noop(self) -> None:
        from backend.v.models.factory import _thinking_extra_body

        assert _thinking_extra_body("deepseek-v4-flash", thinking=False) is None

    def test_get_chat_model_binds_extra_body_for_capable(self) -> None:
        s = LLMSettings(
            _env_file=None,  # type: ignore[call-arg]
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="qwen3.6-max",
            model_fallback="deepseek-v4-flash",
            thinking=True,
        )
        chat = get_chat_model(s, "main_primary")
        assert chat.extra_body == {"enable_thinking": True}

    def test_get_chat_model_no_extra_body_for_incapable(self) -> None:
        s = LLMSettings(
            _env_file=None,  # type: ignore[call-arg]
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="deepseek-v4-flash",
            model_fallback="deepseek-v4-flash",
            thinking=True,  # requested but unsupported → dropped
        )
        chat = get_chat_model(s, "main_primary")
        assert not chat.extra_body  # None or empty, never enable_thinking


class TestGetEmbedding:
    def test_qwen_dimensions_and_model(
        self, llm_settings: LLMSettings, embedding_settings: EmbeddingSettings
    ) -> None:
        emb = get_embedding(llm_settings, embedding_settings)
        assert isinstance(emb, OpenAIEmbeddings)
        assert emb.model == "text-embedding-v4"
        # DashScope rejects the `dimensions` param via the OpenAI-compatible endpoint
        # so factory.get_embedding deliberately omits it (always returns 1024-dim vectors).
        assert emb.dimensions is None
