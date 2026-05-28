"""User-level memory layers (Phase-2 P3 + Phase-3 Group C).

Three temperature tiers:

1. **Session memory (短期)** — :mod:`session_memory`. Redis hash of the
   current session's preferences + observations, TTL = working memory
   TTL (1800s). Disappears with the session.
2. **Event memory (中期)** — :mod:`event_memory` (read API) +
   ``agent.session_memory`` table (write happens in
   :mod:`backend.v.cron.tasks.consolidate_session`). One row per
   completed session; 30-day expiry.
3. **User memory (长期)** — :mod:`long_term` (read) +
   :mod:`memory_extractor` (LLM-driven write). ``agent.user_profile``
   JSONB and ``agent.memory_episodes`` (vectorized). No expiry.

The LangGraph checkpointer (working-memory state, hot/cold migration)
lives separately in :mod:`backend.v.agents.checkpoints` — it is part
of the agent runtime, not user memory.
"""

from backend.v.memory.event_memory import (
    read_recent_event_memories,
    render_recent_events_for_prompt,
)
from backend.v.memory.long_term import read_user_profile
from backend.v.memory.memory_extractor import (
    Episode,
    ExtractionOutput,
    extract_session_memory,
)
from backend.v.memory.session_memory import (
    delete_session_memory,
    read_session_memory,
    write_session_memory,
)
from backend.v.memory.session_memory import (
    render_for_prompt as render_session_memory_for_prompt,
)
from backend.v.memory.working import cache_user_profile, get_cached_user_profile

__all__ = [
    "Episode",
    "ExtractionOutput",
    "cache_user_profile",
    "delete_session_memory",
    "extract_session_memory",
    "get_cached_user_profile",
    "read_recent_event_memories",
    "read_session_memory",
    "read_user_profile",
    "render_recent_events_for_prompt",
    "render_session_memory_for_prompt",
    "write_session_memory",
]
