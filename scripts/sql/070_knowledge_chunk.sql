-- agent.knowledge_chunk: generic semantic-recall corpus.
--
-- One row per searchable knowledge unit. The ``search`` tool embeds the
-- customer's free-text query and returns the top-K rows by cosine
-- distance. ``source_type`` lets a single table host heterogeneous
-- corpora (products, FAQ, return policy, shipping rules…) without
-- forcing each kind into its own schema. ``source_id`` references back
-- to the originating row so the agent can drill into structured data
-- when it needs more than the chunk text exposes.
--
-- ``UNIQUE (source_type, source_id)`` makes seed/refresh scripts
-- idempotent: re-running them upserts in place rather than duplicating.
-- ``tenant_id`` is reserved (NULL today) to keep the multi-tenancy
-- migration non-breaking.

CREATE TABLE IF NOT EXISTS agent.knowledge_chunk (
    chunk_id    BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    tenant_id   TEXT,
    text        TEXT NOT NULL,
    embedding   vector(1024) NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT knowledge_chunk_unique_source UNIQUE (source_type, source_id)
);

-- HNSW over cosine distance. ``vector_cosine_ops`` matches the operator
-- used by the search query (``embedding <=> $1``). m=16 / ef_construction=64
-- are pgvector's default-ish good-quality settings for ≤100k rows.
CREATE INDEX IF NOT EXISTS knowledge_chunk_embedding_hnsw
    ON agent.knowledge_chunk
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Filter pushdown for source_type-scoped recall (e.g., "only FAQ rows").
CREATE INDEX IF NOT EXISTS knowledge_chunk_source_type_idx
    ON agent.knowledge_chunk (source_type);
