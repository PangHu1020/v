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


class LangSmithSettings(BaseSettings):
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


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration. Uses Qwen credentials from ``LLMSettings``."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="EMBEDDING_")

    model: str = "text-embedding-v4"
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
    emotion_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    """Removed emotion preemption — kept field so existing .env files don't error.
    Set to 1.0 to disable (always false). Scheduled for removal."""


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


class WecomAibotSettings(BaseSettings):
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


class MilvusSettings(BaseSettings):
    """Milvus vector database configuration."""

    model_config = SettingsConfigDict(**_COMMON, env_prefix="MILVUS_")

    uri: str = "http://localhost:19530"
    token: str = ""
    collection_name: str = "knowledge_chunks"


class RAGSettings(BaseSettings):
    """RAG cascade search parameters.

    Thresholds derived from eval/ablation on 150-doc corpus (120 products + 30 FAQ):
    - Dense top-1 scores cluster at 0.59-0.82 for noisy/synonym queries.
    - stage1_min=0.88 lets ~80% of hard queries proceed to hybrid (stage-2).
    - stage2_min=0.75 lets remaining hard queries reach LLM-rewrite (stage-3).
    """

    model_config = SettingsConfigDict(**_COMMON, env_prefix="RAG_")

    min_score: float = Field(default=0.76, ge=0.0, le=1.0)
    """Stage-1 (dense cosine) exit threshold → ~60% of queries exit here."""
    stage2_min_score: float = Field(default=0.41, ge=0.0, le=1.0)
    """Stage-2 (hybrid WeightedRanker) exit threshold → ~30% exit here, ~10% fall to rewrite.
    Note: hybrid scores use a different scale (~0.40–0.90) than dense cosine (~0.60–0.95)."""
    min_k: int = Field(default=3, ge=1)
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)


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
    wecom_aibot: WecomAibotSettings
    skill: SkillSettings
    langsmith: LangSmithSettings
    milvus: MilvusSettings
    rag: RAGSettings


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
        wecom_aibot=WecomAibotSettings(),
        skill=SkillSettings(),
        langsmith=LangSmithSettings(),
        milvus=MilvusSettings(),
        rag=RAGSettings(),
    )
