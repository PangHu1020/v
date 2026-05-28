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
    - **用户记忆**（长期，永不过期）—— Postgres ``agent.user_profile``;
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
