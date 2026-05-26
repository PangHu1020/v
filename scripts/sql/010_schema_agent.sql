-- Agent-platform persistence schema.
--
-- Owns: hierarchical memory (working snapshots, session summaries, long-term
-- profile + episodes), session lifecycle metadata, and a placeholder alias
-- table reserved for future cross-channel identity merging.
--
-- Tenant-id columns are nullable in MVP so single-tenant -> multi-tenant is a
-- non-breaking later migration.

CREATE SCHEMA IF NOT EXISTS agent;
SET search_path = agent, public;


-- Reserved for future cross-channel identity merging. Empty in Phase-1.
DROP TABLE IF EXISTS agent.user_alias CASCADE;
CREATE TABLE agent.user_alias (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel           TEXT NOT NULL,
    channel_user_id   TEXT NOT NULL,
    unified_user_id   UUID,
    tenant_id         UUID,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (channel, channel_user_id)
);


-- Per-conversation session state. A new row is minted whenever 30-min silence
-- elapses for a (channel, channel_user_id) pair.
DROP TABLE IF EXISTS agent.session CASCADE;
CREATE TABLE agent.session (
    session_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel           TEXT NOT NULL,
    channel_user_id   TEXT NOT NULL,
    tenant_id         UUID,
    started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_activity_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    status            TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'consolidated', 'suspended', 'closed'))
);

CREATE INDEX idx_session_user
    ON agent.session (channel, channel_user_id, last_activity_at DESC);


-- Consolidated summary written when a session expires or context exceeds a
-- threshold. Source of truth for medium-term recall before long-term distillation.
DROP TABLE IF EXISTS agent.session_memory CASCADE;
CREATE TABLE agent.session_memory (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID NOT NULL REFERENCES agent.session(session_id) ON DELETE CASCADE,
    summary       TEXT NOT NULL,
    token_count   INTEGER NOT NULL DEFAULT 0,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_session_memory_session
    ON agent.session_memory (session_id, created_at DESC);


-- Long-term structured profile. One row per (channel, channel_user_id).
-- Fully injected into the system prompt at on_session_start.
DROP TABLE IF EXISTS agent.user_profile CASCADE;
CREATE TABLE agent.user_profile (
    channel           TEXT NOT NULL,
    channel_user_id   TEXT NOT NULL,
    tenant_id         UUID,
    profile           JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (channel, channel_user_id)
);


-- Long-term episodic memory, vectorized for semantic recall.
-- Embedding dim 1024 matches Qwen text-embedding-v4 (Matryoshka 1024).
-- HNSW index uses cosine ops because the recall_memory tool ranks by
-- recency-weighted cosine similarity.
DROP TABLE IF EXISTS agent.memory_episodes CASCADE;
CREATE TABLE agent.memory_episodes (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel            TEXT NOT NULL,
    channel_user_id    TEXT NOT NULL,
    tenant_id          UUID,
    content            TEXT NOT NULL,
    embedding          vector(1024) NOT NULL,
    metadata           JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_accessed_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_episodes_user
    ON agent.memory_episodes (channel, channel_user_id, created_at DESC);

CREATE INDEX idx_episodes_embedding_hnsw
    ON agent.memory_episodes
    USING hnsw (embedding vector_cosine_ops);
