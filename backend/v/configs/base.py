"""Application-wide settings, loaded from ``.env`` via ``pydantic-settings``.

Each settings class binds to a flat env-var prefix so the ``.env`` file stays
shallow and grep-friendly. The composite :class:`AppSettings` is built by
``get_settings()`` and cached for the process lifetime.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = ".env"
_COMMON = SettingsConfigDict(
    env_file=_ENV_FILE,
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
)


class RuntimeSettings(BaseSettings):
    """Process-level runtime configuration."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="APP_")

    env: str = "dev"
    log_level: str = "INFO"


class LLMSettings(BaseSettings):
    """Provider credentials and routing for chat completions."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="LLM_")

    base_url_deepseek: str = ""
    api_key_deepseek: str = ""
    base_url_qwen: str = ""
    api_key_qwen: str = ""
    main_primary: str = "deepseek-chat-v4-pro"
    main_fallback: str = "deepseek-chat-v4-flash"
    timeout_seconds: int = 30


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration. Uses Qwen credentials from ``LLMSettings``."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="EMBEDDING_")

    model: str = "qwen-text-embedding-v4"
    dim: int = 1024


class DBSettings(BaseSettings):
    """Postgres connection. Single instance with schemas: agent / dw / meta."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="POSTGRES_")

    dsn: str = "postgresql://postgres:postgres@localhost:5432/agent"


class RedisSettings(BaseSettings):
    """Redis connection shared by bus, working memory, and checkpointer."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="REDIS_")

    url: str = "redis://localhost:6379/0"


class MemorySettings(BaseSettings):
    """Memory layer parameters."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="MEMORY_")

    working_ttl_seconds: int = 1800


class BusSettings(BaseSettings):
    """Redis Streams bus configuration."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="BUS_")

    stream_prefix: str = "bus"
    shard_count: int = Field(default=64, ge=1, le=4096)
    consumer_group: str = "workers"
    debounce_ms: int = Field(default=500, ge=0)


class WecomSettings(BaseSettings):
    """WeCom (企业微信) customer-side webhook credentials."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="WECOM_")

    corp_id: str = ""
    agent_id: str = ""
    secret: str = ""
    token: str = ""
    aes_key: str = ""


class FeishuSettings(BaseSettings):
    """Feishu (飞书) customer-side webhook credentials."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="FEISHU_")

    app_id: str = ""
    app_secret: str = ""
    encrypt_key: str = ""
    verification_token: str = ""


class SlackSettings(BaseSettings):
    """Slack operator-side adapter credentials and target channel.

    Phase-2 P0: handoff alerts and operator interactivity. Empty values
    disable the Slack adapter (Phase-1 customer flow continues to work).
    """

    model_config = SettingsConfigDict(**_COMMON, env_prefix="SLACK_")

    bot_token: str = ""
    signing_secret: str = ""
    handoff_channel_id: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.signing_secret and self.handoff_channel_id)


class AppSettings(BaseModel):
    """Composite settings handed to the FastAPI lifespan and to ``/v/`` modules."""

    runtime: RuntimeSettings
    llm: LLMSettings
    embedding: EmbeddingSettings
    db: DBSettings
    redis: RedisSettings
    memory: MemorySettings
    bus: BusSettings
    wecom: WecomSettings
    feishu: FeishuSettings
    slack: SlackSettings


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the cached composite settings instance.

    Settings are loaded once per process. Tests override individual subsections
    by clearing the cache and re-instantiating, or by monkeypatching env vars
    before the first call.
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
        feishu=FeishuSettings(),
        slack=SlackSettings(),
    )
