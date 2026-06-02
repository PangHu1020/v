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
        base_url_deepseek="https://proxy.example.com/deepseek/v1",
        api_key_deepseek="sk-deepseek-test",
        base_url_qwen="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_qwen="sk-qwen-test",
        main_primary="deepseek-chat-v4-pro",
        main_fallback="deepseek-chat-v4-flash",
        timeout_seconds=30,
    )


@pytest.fixture
def embedding_settings() -> EmbeddingSettings:
    return EmbeddingSettings(
        _env_file=None,  # type: ignore[call-arg]
        model="Qwen3-Embedding-0.6B",
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


class TestGetEmbedding:
    def test_qwen_dimensions_and_model(
        self, llm_settings: LLMSettings, embedding_settings: EmbeddingSettings
    ) -> None:
        emb = get_embedding(llm_settings, embedding_settings)
        assert isinstance(emb, OpenAIEmbeddings)
        assert emb.model == "Qwen3-Embedding-0.6B"
        # DashScope rejects the `dimensions` param via the OpenAI-compatible endpoint
        # so factory.get_embedding deliberately omits it (always returns 1024-dim vectors).
        assert emb.dimensions is None
