"""User-level memory: long-term profile + working-memory snapshot.

The LangGraph checkpointer (graph state, hot/cold migration) lives in
``backend.v.agents`` since it is part of agent runtime, not user memory.

This module covers:

- :func:`read_user_profile`: read-only access to the long-term
  ``user_profile`` table. Writes (``memory_extractor``) are Phase-2.
- :func:`cache_user_profile` / :func:`get_cached_user_profile`:
  working-memory Redis snapshot used by ``on_session_start`` to skip a
  Postgres round-trip per turn.
"""

from backend.v.memory.long_term import read_user_profile
from backend.v.memory.working import cache_user_profile, get_cached_user_profile

__all__ = [
    "cache_user_profile",
    "get_cached_user_profile",
    "read_user_profile",
]
