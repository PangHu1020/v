-- Phase-3 Group C: medium-term event memory expiry + index.
--
-- 事件记忆 (medium-term) lives in agent.session_memory with a 30-day TTL.
-- A reaper sweep is not needed in MVP: reads filter by ``expires_at > now()``
-- and the per-row size is small, so let the rows pile up until manual VACUUM
-- decides to reclaim them.

ALTER TABLE agent.session_memory
    ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;

-- Backfill any pre-existing rows with a 30-day forward expiry from now.
UPDATE agent.session_memory
   SET expires_at = now() + INTERVAL '30 days'
 WHERE expires_at IS NULL;

-- Speeds up the on_session_start lookup that pulls the latest few
-- non-expired events for a user. (Predicate uses now() at query time;
-- a partial index can't reference now() because it's STABLE not IMMUTABLE.)
CREATE INDEX IF NOT EXISTS idx_session_memory_recent
    ON agent.session_memory (session_id, expires_at DESC, created_at DESC);
