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


class MCPSettings(BaseSettings):
    """Model Context Protocol client configuration (Phase-2 P1).

    ``servers_json`` is a JSON-encoded list of server configs; each entry
    matches :class:`MCPServerConfig`. Empty list disables MCP entirely.

    Cache TTLs apply to non-write tool results; write-class tools
    (declared per-server in ``write_tools``) skip the cache.
    """

    model_config = SettingsConfigDict(**_COMMON, env_prefix="MCP_")

    servers_json: str = "[]"
    cache_l1_ttl_seconds: int = Field(default=300, ge=0)
    cache_l2_ttl_seconds: int = Field(default=86400, ge=0)
    call_timeout_seconds: int = Field(default=30, ge=1)


class ARQSettings(BaseSettings):
    """ARQ worker settings (Phase-2 P2).

    The ARQ worker is a separate process from the FastAPI app. It runs
    cron tasks (logistics notification, ad-hoc ad push, repurchase
    reminder) and delayed jobs (session memory consolidation). Defaults
    to the same Redis instance as bus / working memory; ``ARQ_REDIS_URL``
    can split queue traffic onto its own database.
    """

    model_config = SettingsConfigDict(**_COMMON, env_prefix="ARQ_")

    redis_url: str = ""
    queue_name: str = "arq:queue"
    max_jobs: int = Field(default=10, ge=1)
    job_timeout_seconds: int = Field(default=120, ge=1)

    def effective_redis_url(self, fallback: str) -> str:
        return self.redis_url or fallback


class SkillSettings(BaseSettings):
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
    mcp: MCPSettings
    arq: ARQSettings
    skill: SkillSettings


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
        mcp=MCPSettings(),
        arq=ARQSettings(),
        skill=SkillSettings(),
    )
