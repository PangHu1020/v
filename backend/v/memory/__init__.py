"""User-level memory layers (Phase-3 Group H reshape).

Three temperature tiers, all sentence-shaped via :class:`MemoryEntry`:

1. **Working memory (短期)** — :mod:`working`. Redis list of
   :class:`MemoryEntry` keyed by ``session_id``. TTL = working memory
   window. Disappears with the session.
2. **Event memory (中期)** — :mod:`event_memory`. PG ``agent.event_memory``
   row-per-entry table, 30-day TTL, ``vector(1024)`` column for
   semantic recall. Read by ``on_session_start`` (newest N) and the
   :func:`backend.v.tools.recall_memory` tool (vector ANN).
3. **User profile (长期)** — :mod:`long_term` (read) +
   :mod:`memory_extractor` (LLM-driven write into ``agent.user_profile``
   JSONB). Schema described by :class:`UserProfile`.

The LangGraph checkpointer (working-memory state, hot/cold migration)
lives separately in :mod:`backend.v.agents.checkpoints` — it is part
of the agent runtime, not user memory.
"""

from backend.v.memory.event_memory import (
    insert_event_memories,
    read_recent_event_memories,
    render_recent_events_for_prompt,
)
from backend.v.memory.long_term import read_user_profile
from backend.v.memory.memory_extractor import (
    promote_to_long_term,
)
from backend.v.memory.prompts import (
    render_session_memory_for_prompt,
)
from backend.v.memory.types import (
    ExtractionResult,
    MemoryEntry,
    MemoryKind,
    UserProfile,
)
from backend.v.memory.working import (
    append_working_memory,
    cache_user_profile,
    delete_working_memory,
    get_cached_user_profile,
    read_working_memory,
)

__all__ = [
    "ExtractionResult",
    "MemoryEntry",
    "MemoryKind",
    "UserProfile",
    "append_working_memory",
    "cache_user_profile",
    "delete_working_memory",
    "get_cached_user_profile",
    "insert_event_memories",
    "promote_to_long_term",
    "read_recent_event_memories",
    "read_user_profile",
    "read_working_memory",
    "render_recent_events_for_prompt",
    "render_session_memory_for_prompt",
]
