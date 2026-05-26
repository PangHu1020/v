-- Postgres extensions required by the agent platform.
-- pgvector: vector(1024) embedding column on agent.memory_episodes.
-- pgcrypto: gen_random_uuid() for surrogate keys.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
