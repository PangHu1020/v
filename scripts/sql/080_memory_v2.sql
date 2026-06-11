-- 080_memory_v2.sql
--
-- Memory system V2: split the flat "everything is a MemoryEntry sentence"
-- model into two clearly-bounded stores, and wire forgetting via monthly
-- consolidation instead of TTL-based invisibility.
--
-- Two tiers touched here (working memory stays in Redis, untouched):
--
--   agent.user_memory   NEW. Bi-temporal, overwritable attributes about the
--                       customer (preferences / constraints / behavioural
--                       patterns). Has a single "current truth" per attr_key
--                       and an append-only supersession chain for audit.
--                       No embedding — small, stable, injected whole at
--                       session start.
--
--   agent.event_memory  EXTENDED (ALTER, not DROP — recall_memory.py queries
--                       it live). Append-only episodic events. New columns
--                       support subject-anchored consolidation, ACT-R
--                       activation, source/confidence provenance, and the
--                       raw→summary tier downgrade that bounds growth.
--
-- Rerun-safe: every statement uses IF NOT EXISTS / ADD COLUMN IF NOT EXISTS.

SET search_path = agent, public;


-- ── User memory (bi-temporal, overwritable) ──────────────────────────────────
--
-- Conflict model:
--   * A key has at most ONE active row (the partial unique index enforces it).
--   * Overwriting an attribute closes the old row (status='superseded',
--     valid_to=now(), superseded_by=<new id>) and inserts a new active row.
--     Old rows are never deleted — the chain IS the audit log.
--   * Arbitration when a new candidate conflicts with the active value:
--     stated > inferred, then higher confidence, then most-recent wins.
--     (Enforced in the write pipeline, not in SQL.)
CREATE TABLE IF NOT EXISTS agent.user_memory (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel           TEXT NOT NULL,
    channel_user_id   TEXT NOT NULL,
    tenant_id         TEXT,

    -- Attribute identity. attr_key is the conflict key; attr_value is a JSONB
    -- scalar/array so both canonical fields and open long-tail keys share one
    -- physical shape and both get the same audit treatment.
    attr_key          TEXT NOT NULL,
    attr_value        JSONB NOT NULL,
    kind              TEXT NOT NULL
        CHECK (kind IN ('preference', 'constraint', 'pattern')),

    -- Provenance: did the customer state this, or did the model infer it?
    -- Drives conflict arbitration and whether the agent may repeat it back.
    source            TEXT NOT NULL DEFAULT 'inferred'
        CHECK (source IN ('stated', 'inferred')),
    confidence        REAL NOT NULL DEFAULT 0.5
        CHECK (confidence >= 0.0 AND confidence <= 1.0),

    -- Bi-temporal validity. valid_to IS NULL while the row is the current
    -- truth; status mirrors it for cheap filtering / index predicate.
    status            TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'superseded')),
    valid_from        TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to          TIMESTAMPTZ,
    superseded_by     UUID REFERENCES agent.user_memory(id) ON DELETE SET NULL,

    session_id        UUID REFERENCES agent.session(session_id) ON DELETE SET NULL,
    -- When this attribute's value was last reaffirmed by the customer. The
    -- recall/injection layer can flag a value as "possibly stale" without
    -- forgetting it (semantic memory never decays on disuse).
    last_confirmed_at TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE agent.user_memory IS
    'Bi-temporal, overwritable customer attributes (preference/constraint/pattern). '
    'One active row per attr_key; superseded rows form the audit chain.';
COMMENT ON COLUMN agent.user_memory.source IS
    'stated = customer said it; inferred = model deduced it. Arbitration + repeat-back gate.';
COMMENT ON COLUMN agent.user_memory.valid_to IS
    'NULL while current. Set to now() when superseded — the row is kept, not deleted.';

-- At most one active value per attribute per customer. This is the conflict
-- key: an overwrite must close the old active row before inserting the new.
CREATE UNIQUE INDEX IF NOT EXISTS user_memory_active_unique
    ON agent.user_memory (channel, channel_user_id, attr_key)
    WHERE status = 'active';

-- Audit-chain lookups: "show me the full history of attr_key for this user".
CREATE INDEX IF NOT EXISTS user_memory_history_idx
    ON agent.user_memory (channel, channel_user_id, attr_key, valid_from DESC);


-- ── Episodic memory extensions (append-only events) ──────────────────────────
--
-- The base table (id, channel, channel_user_id, session_id, content, kind,
-- importance, keywords, embedding, created_at, expires_at, last_accessed_at)
-- was created in 060_memory_reshape.sql and is queried live by
-- recall_memory.py. We ADD columns only — no drops, no renames.
ALTER TABLE agent.event_memory
    -- Entity / topic anchor (order id, SKU, "尺码咨询"). Drives both dedup
    -- (same subject + high cosine = merge) and monthly per-subject summaries.
    ADD COLUMN IF NOT EXISTS subject TEXT,
    -- Provenance, mirroring user_memory.
    ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'inferred'
        CHECK (source IN ('stated', 'inferred')),
    ADD COLUMN IF NOT EXISTS confidence REAL NOT NULL DEFAULT 0.5
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    -- raw = original event; summary = monthly per-subject consolidation that
    -- absorbed a cluster of raws.
    ADD COLUMN IF NOT EXISTS tier TEXT NOT NULL DEFAULT 'raw'
        CHECK (tier IN ('raw', 'summary')),
    -- Owning month for summaries / the month a raw belongs to, e.g. '2026-06'.
    ADD COLUMN IF NOT EXISTS period TEXT,
    -- Reinforcement counter: bumped by recall_memory on every hit ("用进").
    ADD COLUMN IF NOT EXISTS access_count INTEGER NOT NULL DEFAULT 0,
    -- Materialised ACT-R base-level activation, refreshed by the consolidation
    -- scan: importance + ln(1+access_count) − ln(1+age_days). Low activation
    -- makes a raw eligible for deletion after its month is summarised.
    ADD COLUMN IF NOT EXISTS activation REAL,
    -- Which summary row absorbed this raw (NULL = not yet consolidated).
    ADD COLUMN IF NOT EXISTS consolidated_into UUID
        REFERENCES agent.event_memory(id) ON DELETE SET NULL;

COMMENT ON COLUMN agent.event_memory.subject IS
    'Entity/topic anchor for dedup clustering and per-subject monthly summaries.';
COMMENT ON COLUMN agent.event_memory.tier IS
    'raw = original event; summary = monthly per-subject consolidation.';
COMMENT ON COLUMN agent.event_memory.activation IS
    'ACT-R base-level activation; drives use-it-or-lose-it raw eviction at consolidation.';
COMMENT ON COLUMN agent.event_memory.expires_at IS
    'Legacy TTL filter (recall still honours it). V2 writes leave it NULL — '
    'forgetting is driven by monthly consolidation, not TTL.';

-- Consolidation scan: find this user's raw rows in a closed period, grouped by
-- subject. Partial index keeps it tight (only un-consolidated raws matter).
CREATE INDEX IF NOT EXISTS event_memory_consolidation_idx
    ON agent.event_memory (channel, channel_user_id, period, subject)
    WHERE tier = 'raw' AND consolidated_into IS NULL;
