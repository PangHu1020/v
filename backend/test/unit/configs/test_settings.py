"""Tests for ``backend.v.configs.base``."""

from __future__ import annotations

import pytest

from backend.v.configs.base import (
    AppSettings,
    BusSettings,
    EmbeddingSettings,
    LLMSettings,
    MemorySettings,
    RAGSettings,
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
        monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
        monkeypatch.setenv("LLM_MODEL_FALLBACK", "deepseek-reasoner")
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "60")
        settings = _no_env_file(LLMSettings)
        assert settings.model == "deepseek-chat"
        assert settings.model_fallback == "deepseek-reasoner"
        assert settings.timeout_seconds == 60

    def test_provider_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-primary")
        monkeypatch.setenv("LLM_BASE_URL_FALLBACK", "https://api.deepseek.com/v1")
        monkeypatch.setenv("LLM_API_KEY_FALLBACK", "sk-fallback")
        settings = _no_env_file(LLMSettings)
        assert settings.base_url == "https://api.siliconflow.cn/v1"
        assert settings.api_key == "sk-primary"
        assert settings.base_url_fallback == "https://api.deepseek.com/v1"
        assert settings.api_key_fallback == "sk-fallback"

    def test_thinking_defaults_false_and_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _no_env_file(LLMSettings).thinking is False
        monkeypatch.setenv("LLM_THINKING", "true")
        assert _no_env_file(LLMSettings).thinking is True


class TestEmbeddingSettings:
    def test_defaults_match_locked_decision(self) -> None:
        settings = _no_env_file(EmbeddingSettings)
        assert settings.model == "text-embedding-v4"
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

    def test_v2_defaults(self) -> None:
        s = _no_env_file(MemorySettings)
        assert s.consolidation_importance_floor == 0.3
        assert s.consolidation_dedup_similarity_floor == 0.92
        assert s.activation_w_importance == 1.0
        assert s.activation_w_access == 0.5
        assert s.activation_w_age == 0.3
        assert s.consolidation_survival_threshold == 0.5
        assert s.consolidation_min_cluster_size == 2

    def test_v2_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_CONSOLIDATION_SURVIVAL_THRESHOLD", "0.7")
        monkeypatch.setenv("MEMORY_ACTIVATION_W_ACCESS", "0.9")
        s = _no_env_file(MemorySettings)
        assert s.consolidation_survival_threshold == 0.7
        assert s.activation_w_access == 0.9


class TestIntentSettings:
    def test_defaults(self) -> None:
        from backend.v.configs.base import IntentSettings

        s = _no_env_file(IntentSettings)
        assert s.threshold_refund == 0.6
        assert s.threshold_chitchat == 0.5
        assert s.ambiguity_margin == 0.15
        assert s.max_clarify_turns == 1

    def test_threshold_for_lookup(self) -> None:
        from backend.v.configs.base import IntentSettings

        s = _no_env_file(IntentSettings)
        assert s.threshold_for("logistics") == 0.55
        # Unknown intent falls back to the general threshold.
        assert s.threshold_for("bogus") == s.threshold_general

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from backend.v.configs.base import IntentSettings

        monkeypatch.setenv("INTENT_THRESHOLD_REFUND", "0.8")
        monkeypatch.setenv("INTENT_MAX_CLARIFY_TURNS", "2")
        s = _no_env_file(IntentSettings)
        assert s.threshold_refund == 0.8
        assert s.max_clarify_turns == 2


class TestChannelSettings:
    def test_wecom_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WECOM_CORP_ID", "ww123")
        monkeypatch.setenv("WECOM_AES_KEY", "abc" * 14)
        settings = _no_env_file(WecomSettings)
        assert settings.corp_id == "ww123"
        assert settings.aes_key == "abc" * 14


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


class TestYamlConfig:
    """YAML config layer: reflection-sectioned source, env-override precedence.

    These construct settings with ``_env_file=None`` so the on-disk ``.env``
    (which carries real values like ``BUS_SHARD_COUNT``) doesn't mask the YAML
    layer under test. Precedence proven here: env var > YAML > code default.
    """

    def _write_yaml(self, tmp_path, body: str):
        from backend.v.configs.base import _load_yaml

        p = tmp_path / "config.yaml"
        p.write_text(body, encoding="utf-8")
        _load_yaml.cache_clear()  # path-keyed cache; clear so the new file is read
        return p

    def test_yaml_overrides_code_default(self, tmp_path, monkeypatch) -> None:
        p = self._write_yaml(
            tmp_path,
            "rag:\n  rerank_candidates: 50\n  min_k: 7\nbus:\n  shard_count: 128\n",
        )
        monkeypatch.setenv("APP_CONFIG_FILE", str(p))
        # Clear any ambient env vars so we test the YAML > default layer.
        for k in ("RAG_RERANK_CANDIDATES", "RAG_MIN_K", "BUS_SHARD_COUNT"):
            monkeypatch.delenv(k, raising=False)
        assert RAGSettings(_env_file=None).rerank_candidates == 50
        assert RAGSettings(_env_file=None).min_k == 7
        assert BusSettings(_env_file=None).shard_count == 128

    def test_env_overrides_yaml(self, tmp_path, monkeypatch) -> None:
        p = self._write_yaml(tmp_path, "rag:\n  rerank_candidates: 50\n")
        monkeypatch.setenv("APP_CONFIG_FILE", str(p))
        monkeypatch.setenv("RAG_RERANK_CANDIDATES", "99")
        # env var wins over YAML
        assert RAGSettings(_env_file=None).rerank_candidates == 99

    def test_missing_section_falls_back_to_default(self, tmp_path, monkeypatch) -> None:
        p = self._write_yaml(tmp_path, "rag:\n  rerank_candidates: 50\n")
        monkeypatch.setenv("APP_CONFIG_FILE", str(p))
        monkeypatch.delenv("LLM_MODEL", raising=False)
        # llm section absent from YAML → code defaults
        assert LLMSettings(_env_file=None).model == "deepseek-chat-v4-pro"

    def test_missing_file_is_noop(self, tmp_path, monkeypatch) -> None:
        from backend.v.configs.base import _load_yaml

        _load_yaml.cache_clear()
        monkeypatch.setenv("APP_CONFIG_FILE", str(tmp_path / "does-not-exist.yaml"))
        monkeypatch.delenv("RAG_RERANK_CANDIDATES", raising=False)
        assert RAGSettings(_env_file=None).rerank_candidates == 20  # code default

    def test_yaml_respects_validation(self, tmp_path, monkeypatch) -> None:
        # shard_count has ge=1, le=4096 — an out-of-range YAML value must raise
        p = self._write_yaml(tmp_path, "bus:\n  shard_count: 99999\n")
        monkeypatch.setenv("APP_CONFIG_FILE", str(p))
        monkeypatch.delenv("BUS_SHARD_COUNT", raising=False)
        with pytest.raises(ValueError):
            BusSettings(_env_file=None)
