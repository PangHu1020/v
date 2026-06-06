"""Application-wide settings, loaded by reflection from a YAML config file
and ``.env`` via ``pydantic-settings``.

Two layers, by design:

- **``config.yaml``** (path from ``APP_CONFIG_FILE``, default ``./config.yaml``):
  non-secret tunables — model names, thresholds, TTLs, shard counts.
  Edit this to change behaviour without touching code.
- **``.env`` / environment variables**: secrets and deployment overrides —
  API keys, DSNs, bot credentials.

Precedence (highest wins): **env var > .env > config.yaml > code default**.
So a YAML value overrides the hard-coded default, and an env var overrides
the YAML value. Secrets therefore stay out of the committed YAML.

The YAML file is *sectioned by reflection*: each sub-settings class maps to a
top-level YAML key derived from its field name on :class:`AppSettings`
(``LLMSettings`` → ``llm``, ``RAGSettings`` → ``rag`` …). No hand-written
section table — the field names on ``AppSettings`` are the single source of
truth, so adding a new settings class needs no extra wiring here.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

_ENV_FILE = ".env"
_CONFIG_FILE_ENV = "APP_CONFIG_FILE"
_DEFAULT_CONFIG_FILE = "config.yaml"

_COMMON = SettingsConfigDict(
    env_file=_ENV_FILE,
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
)


@lru_cache(maxsize=8)
def _load_yaml(path_str: str) -> dict[str, Any]:
    """Parse a YAML config file into a dict (cached). Missing file → ``{}``."""
    path = Path(path_str)
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _config_path() -> str:
    return os.environ.get(_CONFIG_FILE_ENV, _DEFAULT_CONFIG_FILE)


# Reverse map: sub-settings class -> its YAML section name.
# Populated lazily after AppSettings is defined (see _section_for).
_SECTION_CACHE: dict[type, str | None] = {}


def _section_for(cls: type) -> str | None:
    """Return the YAML top-level key for a settings class, via AppSettings reflection.

    ``AppSettings.model_fields`` maps a field name (the YAML section) to the
    annotated sub-settings type. We invert it once and cache. ``None`` when the
    class is not part of the composite (e.g. instantiated standalone in a test).
    """
    if cls in _SECTION_CACHE:
        return _SECTION_CACHE[cls]
    section: str | None = None
    # AppSettings is defined at module load, below; guard for first-call ordering.
    app_cls = globals().get("AppSettings")
    if app_cls is not None:
        for field_name, field in app_cls.model_fields.items():
            if field.annotation is cls:
                section = field_name
                break
    _SECTION_CACHE[cls] = section
    return section


class SectionedYamlSource(PydanticBaseSettingsSource):
    """Feed one section of the YAML config file into a settings class.

    The section is resolved by reflection from ``AppSettings`` field names.
    Only keys present in the YAML are returned; everything else falls through
    to lower-priority sources (defaults).
    """

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # Not used — we override __call__ to return the whole section at once.
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        section = _section_for(self.settings_cls)
        if section is None:
            return {}
        data = _load_yaml(_config_path())
        block = data.get(section)
        return dict(block) if isinstance(block, dict) else {}


class _YamlSettings(BaseSettings):
    """Base for all sub-settings: layers the sectioned YAML source under env vars."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Order = priority (first wins): init > env > .env > YAML > file secrets.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            SectionedYamlSource(settings_cls),
            file_secret_settings,
        )


class RuntimeSettings(_YamlSettings):
    """Process-level runtime configuration."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="APP_")

    env: str = "dev"
    log_level: str = "INFO"


class LLMSettings(_YamlSettings):
    """Provider credentials and routing for chat completions."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="LLM_")

    base_url_deepseek: str = ""
    api_key_deepseek: str = ""
    base_url_qwen: str = ""
    api_key_qwen: str = ""
    main_primary: str = "deepseek-chat-v4-pro"
    main_fallback: str = "deepseek-chat-v4-flash"
    timeout_seconds: int = 30


class LangSmithSettings(_YamlSettings):
    """LangSmith tracing configuration.

    When ``api_key`` is set and ``tracing`` is true, every LangChain /
    LangGraph invocation in this process emits a trace to the LangSmith
    project named by ``project``. The integration is opt-in: with no key
    set, nothing is sent and there is zero runtime overhead.
    """

    model_config = SettingsConfigDict(**_COMMON, env_prefix="LANGSMITH_")

    tracing: bool = False
    api_key: str = ""
    project: str = "v-main"
    endpoint: str = "https://api.smith.langchain.com"


class EmbeddingSettings(_YamlSettings):
    """Embedding model configuration. Uses Qwen credentials from ``LLMSettings``."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="EMBEDDING_")

    model: str = "text-embedding-v4"
    dim: int = 1024


class DBSettings(_YamlSettings):
    """Postgres connection. Single instance with schemas: agent / dw / meta."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="POSTGRES_")

    dsn: str = "postgresql://postgres:postgres@localhost:5432/agent"


class RedisSettings(_YamlSettings):
    """Redis connection shared by bus, working memory, and checkpointer."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="REDIS_")

    url: str = "redis://localhost:6379/0"


class MemorySettings(_YamlSettings):
    """Memory layer parameters (Phase-3 三层温度梯度).

    The three layers and their levers:

    - **会话记忆**（短期，本会话内）—— Redis hash; TTL = working_ttl_seconds.
      Holds ``preferences`` + ``observations`` extracted from this
      session's messages; written by mid-session compression and the
      session-expiry trigger; consumed by the session-end promotion to
      user_profile.
    - **事件记忆**（中期，event_ttl_days 内）—— Postgres ``agent.session_memory``
      rows with ``expires_at = now() + event_ttl_days * day``; the most
      recent ``recent_events_to_inject`` rows are auto-injected at
      ``on_session_start``.
    - **用户记忆**（长期，���不过期）—— Postgres ``agent.user_profile``;
      injected in full at ``on_session_start``.

    ``compression_threshold_tokens`` triggers the mid-session
    compression node: when ``messages`` total tokens exceed it, the
    older messages are replaced with the just-written event-memory
    summary so the prompt stays bounded.
    """

    model_config = SettingsConfigDict(**_COMMON, env_prefix="MEMORY_")

    working_ttl_seconds: int = Field(default=1800, ge=60)
    event_ttl_days: int = Field(default=30, ge=1)
    recent_events_to_inject: int = Field(default=3, ge=0, le=20)
    compression_threshold_tokens: int = Field(default=4000, ge=500)
    compression_keep_recent_messages: int = Field(default=4, ge=1)
    """When compression fires, how many trailing messages to keep verbatim
    (so the agent still has the immediate exchange in full fidelity)."""
    emotion_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    """Removed emotion preemption — kept field so existing .env files don't error.
    Set to 1.0 to disable (always false). Scheduled for removal."""


class BusSettings(_YamlSettings):
    """Redis Streams bus configuration."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="BUS_")

    stream_prefix: str = "bus"
    shard_count: int = Field(default=64, ge=1, le=4096)
    consumer_group: str = "workers"
    debounce_ms: int = Field(default=500, ge=0)


class WecomSettings(_YamlSettings):
    """WeCom (企业微信) customer-side webhook credentials."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="WECOM_")

    corp_id: str = ""
    agent_id: str = ""
    secret: str = ""
    token: str = ""
    aes_key: str = ""


class WecomAibotSettings(_YamlSettings):
    """WeCom 智能机器人 (WebSocket) adapter settings (Phase-3 Group G).

    Unlike the HTTP webhook ``WecomSettings``, the 智能机器人 adapter
    runs as a standalone worker process that holds a single persistent
    WebSocket. Outbound deliveries are funnelled through a Redis pub/sub
    channel so any process (FastAPI app, ARQ worker, Slack handoff) can
    publish a reply and the worker forwards it over its WS.

    Auth is the WeCom 智能机器人 ``aibot_subscribe`` handshake:
    ``{bot_id, secret}`` posted as the first frame after connect — NOT
    an HTTP ``Authorization`` header.
    """

    model_config = SettingsConfigDict(**_COMMON, env_prefix="WECOM_AIBOT_")

    ws_url: str = ""
    bot_id: str = ""
    secret: str = ""
    heartbeat_seconds: int = Field(default=30, ge=1)
    outbound_pubsub_channel: str = "wecom_aibot:outbound"


class SkillSettings(_YamlSettings):
    """Skill loader settings (Phase-2 P4).

    Skills are markdown SOPs the agent can prepend to its system prompt
    based on the customer's intent. Phase-2 P4 ships markdown-only;
    SKILL.md+exec and community-registry sources are Phase-3.
    """

    model_config = SettingsConfigDict(**_COMMON, env_prefix="SKILL_")

    internal_repo_path: str = ""
    """Filesystem path to the directory holding markdown skills. Empty
    disables the loader."""

    max_skills_per_turn: int = Field(default=3, ge=0, le=20)
    """Top-K skills to inject per matched turn. 0 disables injection."""


class MilvusSettings(_YamlSettings):
    """Milvus vector database configuration."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="MILVUS_")

    uri: str = "http://localhost:19530"
    token: str = ""
    collection_name: str = "knowledge_chunks"


class RAGSettings(_YamlSettings):
    """RAG cascade search parameters.

    Thresholds derived from eval/ablation on a 150-doc corpus:
    - Stage-1 dense cosine exit at ``min_score`` (~60% of queries).
    - Stage-2 hybrid WeightedRanker exit at ``stage2_min_score`` (~30%).
    - Remaining ~10% reach the Stage-3 LLM rewrite.
    Note hybrid scores live on a different scale (~0.40–0.90) than dense
    cosine (~0.60–0.95); tune the two thresholds independently in config.yaml.
    """

    model_config = SettingsConfigDict(**_COMMON, env_prefix="RAG_")

    min_score: float = Field(default=0.76, ge=0.0, le=1.0)
    stage2_min_score: float = Field(default=0.41, ge=0.0, le=1.0)
    min_k: int = Field(default=3, ge=1)
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class AppSettings(BaseModel):
    """Composite settings handed to the FastAPI lifespan and to ``/v/`` modules.

    Field names here ARE the YAML section names (resolved by reflection in
    :func:`_section_for`). Keep them in sync with ``config.example.yaml``.
    """

    runtime: RuntimeSettings
    llm: LLMSettings
    embedding: EmbeddingSettings
    db: DBSettings
    redis: RedisSettings
    memory: MemorySettings
    bus: BusSettings
    wecom: WecomSettings
    wecom_aibot: WecomAibotSettings
    skill: SkillSettings
    langsmith: LangSmithSettings
    milvus: MilvusSettings
    rag: RAGSettings


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the cached composite settings instance.

    Each sub-settings is built independently, layering env > .env > YAML >
    default. Settings are read once per process; tests clear this cache (see
    ``backend/test/conftest.py``) and point ``APP_CONFIG_FILE`` away so they
    run against pure code defaults.
    """
    return AppSettings(
        runtime=RuntimeSettings(),
        llm=LLMSettings(),
        embedding=EmbeddingSettings(),
        db=DBSettings(),
        redis=RedisSettings(),
        memory=MemorySettings(),
        bus=BusSettings(),
        wecom=WecomSettings(),
        wecom_aibot=WecomAibotSettings(),
        skill=SkillSettings(),
        langsmith=LangSmithSettings(),
        milvus=MilvusSettings(),
        rag=RAGSettings(),
    )
