"""Tests for ``backend.v.configs.base``."""

from __future__ import annotations

import pytest

from backend.v.configs.base import (
    AppSettings,
    BusSettings,
    EmbeddingSettings,
    FeishuSettings,
    LLMSettings,
    MemorySettings,
    RuntimeSettings,
    WecomSettings,
    get_settings,
)


def _no_env_file(cls):
    """Instantiate a settings class without reading the on-disk ``.env`` file."""
    return cls(_env_file=None)  # type: ignore[call-arg]


class TestRuntimeSettings:
    def test_defaults(self) -> None:
        settings = _no_env_file(RuntimeSettings)
        assert settings.env == "dev"
        assert settings.log_level == "INFO"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "prod")
        monkeypatch.setenv("APP_LOG_LEVEL", "WARNING")
        settings = _no_env_file(RuntimeSettings)
        assert settings.env == "prod"
        assert settings.log_level == "WARNING"

    def test_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("app_env", "staging")
        settings = _no_env_file(RuntimeSettings)
        assert settings.env == "staging"


class TestLLMSettings:
    def test_prefix_routes_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_MAIN_PRIMARY", "deepseek-chat")
        monkeypatch.setenv("LLM_MAIN_FALLBACK", "deepseek-reasoner")
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "60")
        settings = _no_env_file(LLMSettings)
        assert settings.main_primary == "deepseek-chat"
        assert settings.main_fallback == "deepseek-reasoner"
        assert settings.timeout_seconds == 60

    def test_provider_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_BASE_URL_DEEPSEEK", "https://api.deepseek.com/v1")
        monkeypatch.setenv("LLM_API_KEY_DEEPSEEK", "sk-test")
        monkeypatch.setenv("LLM_BASE_URL_QWEN", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        monkeypatch.setenv("LLM_API_KEY_QWEN", "sk-qwen")
        settings = _no_env_file(LLMSettings)
        assert settings.base_url_deepseek == "https://api.deepseek.com/v1"
        assert settings.api_key_deepseek == "sk-test"
        assert settings.base_url_qwen.endswith("/compatible-mode/v1")
        assert settings.api_key_qwen == "sk-qwen"


class TestEmbeddingSettings:
    def test_defaults_match_locked_decision(self) -> None:
        settings = _no_env_file(EmbeddingSettings)
        assert settings.model == "qwen-text-embedding-v4"
        assert settings.dim == 1024


class TestBusSettings:
    def test_defaults(self) -> None:
        settings = _no_env_file(BusSettings)
        assert settings.stream_prefix == "bus"
        assert settings.shard_count == 64
        assert settings.consumer_group == "workers"
        assert settings.debounce_ms == 500

    def test_shard_count_lower_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BUS_SHARD_COUNT", "0")
        with pytest.raises(ValueError):
            _no_env_file(BusSettings)

    def test_shard_count_upper_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BUS_SHARD_COUNT", "10000")
        with pytest.raises(ValueError):
            _no_env_file(BusSettings)


class TestMemorySettings:
    def test_default_ttl(self) -> None:
        settings = _no_env_file(MemorySettings)
        assert settings.working_ttl_seconds == 1800


class TestChannelSettings:
    def test_wecom_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WECOM_CORP_ID", "ww123")
        monkeypatch.setenv("WECOM_AES_KEY", "abc" * 14)
        settings = _no_env_file(WecomSettings)
        assert settings.corp_id == "ww123"
        assert settings.aes_key == "abc" * 14

    def test_feishu_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FEISHU_APP_ID", "cli_app")
        monkeypatch.setenv("FEISHU_VERIFICATION_TOKEN", "tok")
        settings = _no_env_file(FeishuSettings)
        assert settings.app_id == "cli_app"
        assert settings.verification_token == "tok"


class TestAppSettings:
    def test_get_settings_returns_composite(self) -> None:
        settings = get_settings()
        assert isinstance(settings, AppSettings)
        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.bus, BusSettings)

    def test_get_settings_is_cached(self) -> None:
        first = get_settings()
        second = get_settings()
        assert first is second
