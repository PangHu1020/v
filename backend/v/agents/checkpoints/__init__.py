"""Checkpoint storage for LangGraph state.

Two implementations sharing :class:`AsyncCheckpointer` Protocol:

- :class:`RedisCheckpointer` (hot path) — short-TTL Redis hash, used by
  the bus worker on every reactive turn.
- :func:`open_pg_checkpointer` (cold path) — durable Postgres saver from
  ``langgraph-checkpoint-postgres``, used while a thread is suspended
  for human handoff.

Hot/cold migration helpers in :mod:`migration` move a thread's full
checkpoint chain between the two stores when ``transfer_to_human``
fires or when the operator clicks "Resume AI".
"""

from backend.v.agents.checkpoints.migration import (
    migrate_cold_to_hot,
    migrate_hot_to_cold,
)
from backend.v.agents.checkpoints.postgres import open_pg_checkpointer
from backend.v.agents.checkpoints.protocol import AsyncCheckpointer
from backend.v.agents.checkpoints.redis import RedisCheckpointer

__all__ = [
    "AsyncCheckpointer",
    "RedisCheckpointer",
    "migrate_cold_to_hot",
    "migrate_hot_to_cold",
    "open_pg_checkpointer",
]
