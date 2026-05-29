-- 060_memory_reshape.sql
--
-- Phase-3 Group H: collapse the medium-term and long-term sentence
-- stores into a single uniform shape (MemoryEntry: content + metadata
-- + embedding) and drop the multi-field session_memory body.
--
-- Old layout:
--   agent.session_memory   one-row-per-session, multi-field summary
--                          (narrative + intents + key_facts + sentiment
--                          + unresolved). Plain TEXT, no embedding.
--   agent.memory_episodes  separate long-term sentence list, vector(1024).
--
-- New layout:
--   agent.event_memory     one-row-per-sentence, vector(1024) on every
--                          row, 30-day TTL via expires_at. Replaces
--                          BOTH old tables. The recall_memory tool
--                          queries here instead of memory_episodes.
--
-- Rerun-safe: every DROP is IF EXISTS / CASCADE.

-- Drop the two predecessors. Their data (consolidated session
-- summaries + episode sentences) is not retained — the Phase-3 reshape
-- is incompatible. In dev this is fine; for any future production
-- migration, write a one-shot ETL that maps old narrative + key_facts
-- onto the new MemoryEntry rows before running this script.
DROP TABLE IF EXISTS agent.session_memory CASCADE;
DROP TABLE IF EXISTS agent.memory_episodes CASCADE;

-- One row per memory sentence. Same physical shape backs both the
-- "recent N events" injection (read-time, ORDER BY created_at DESC)
-- and the recall_memory tool (read-time, vector ANN + recency decay).
CREATE TABLE agent.event_memory (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel         TEXT NOT NULL,
    channel_user_id TEXT NOT NULL,
    -- Optional: NULL when a write doesn't belong to a tracked session
    -- (e.g., a backfill). ON DELETE SET NULL so deleting a session
    -- doesn't lose the historical fact.
    session_id      UUID REFERENCES agent.session(session_id) ON DELETE SET NULL,

    content         TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('preference', 'observation', 'event')),
    importance      REAL NOT NULL CHECK (importance >= 0.0 AND importance <= 1.0),
    keywords        TEXT[] NOT NULL DEFAULT '{}',

    embedding       vector(1024) NOT NULL,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- NULL = no TTL. Default consolidation populates this with
    -- ``now() + 30 days``; long-tail facts the model marks as critical
    -- can be written with NULL to live indefinitely.
    expires_at      TIMESTAMPTZ,
    -- Touched by the recall_memory tool so future maintenance jobs can
    -- prune never-recalled entries if needed.
    last_accessed_at TIMESTAMPTZ
);

COMMENT ON TABLE agent.event_memory IS
    'Phase-3 unified medium-term + long-term sentence store. Replaces session_memory + memory_episodes.';
COMMENT ON COLUMN agent.event_memory.kind IS
    'preference | observation | event — set by the LLM extractor.';
COMMENT ON COLUMN agent.event_memory.importance IS
    '0.0 = trivia, 1.0 = critical (used as a recall-rerank multiplier).';
COMMENT ON COLUMN agent.event_memory.expires_at IS
    'NULL = no TTL. 30-day default for ordinary entries.';

-- Read patterns:
--   1) Recent-N injection: WHERE channel/user filter, ORDER BY created_at DESC, LIMIT N.
--   2) recall_memory tool:  vector ANN, recency-weighted re-rank.
CREATE INDEX idx_event_memory_user_recent
    ON agent.event_memory (channel, channel_user_id, created_at DESC);

CREATE INDEX idx_event_memory_embedding_hnsw
    ON agent.event_memory
    USING hnsw (embedding vector_cosine_ops);

-- Periodic prune query lives outside this script; the column makes the
-- intent self-documenting.
CREATE INDEX idx_event_memory_expires
    ON agent.event_memory (expires_at)
    WHERE expires_at IS NOT NULL;
