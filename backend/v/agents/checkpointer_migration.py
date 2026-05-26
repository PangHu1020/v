"""Hot/cold migration between Redis (working memory) and Postgres (durable).

Triggered by hooks around the ``transfer_to_human`` interrupt:

- ``on_interrupt`` -> :func:`migrate_hot_to_cold`: copy the entire thread's
  checkpoint history from Redis to Postgres, then drop the Redis hash so
  it cannot expire under TTL while the operator is reading.
- ``on_resume`` (Slack button click) -> :func:`migrate_cold_to_hot`: pull
  the thread back into Redis so the live worker can resume the graph
  against the hot path again.

Both operations are idempotent: a partial run (e.g., process crashes
mid-migration) leaves the thread reachable in *at least one* store; the
next run cleans up. The cost of a duplicate checkpoint is negligible.
"""

from __future__ import annotations

from typing import Any, Protocol

from backend.v.agents.checkpointer import RedisCheckpointer
from backend.v.utils.logging import get_logger

_log = get_logger("memory.migration")


class _AsyncCheckpointer(Protocol):
    """Subset of BaseCheckpointSaver we use here. Lets us avoid a hard
    import dependency on AsyncPostgresSaver in tests that use stubs."""

    async def aput(self, config, checkpoint, metadata, new_versions): ...
    async def aput_writes(self, config, writes, task_id, task_path=""): ...
    async def aget_tuple(self, config): ...
    def alist(self, config, *, filter=None, before=None, limit=None): ...
    async def adelete_thread(self, thread_id: str) -> None: ...


def _config(thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": ""}
    if checkpoint_id:
        cfg["checkpoint_id"] = checkpoint_id
    return {"configurable": cfg}


async def _replay(
    *,
    src: _AsyncCheckpointer,
    dst: _AsyncCheckpointer,
    thread_id: str,
) -> int:
    """Replay all checkpoints for ``thread_id`` from ``src`` into ``dst``."""
    config = _config(thread_id)
    tuples = []
    async for tup in src.alist(config):
        tuples.append(tup)
    # ``alist`` returns newest first; replay oldest-to-newest so parent
    # references stay consistent in the destination store. Each ``aput``
    # takes the *parent* config so the new checkpoint's parent pointer is
    # set correctly; passing ``tup.config`` would store the checkpoint as
    # its own parent.
    for tup in reversed(tuples):
        parent_cfg = tup.parent_config or _config(thread_id)
        await dst.aput(parent_cfg, tup.checkpoint, tup.metadata, {})
    return len(tuples)


async def migrate_hot_to_cold(
    thread_id: str,
    *,
    redis_ckpt: RedisCheckpointer,
    pg_ckpt: _AsyncCheckpointer,
) -> int:
    """Move a thread from Redis (hot) to Postgres (cold).

    Args:
        thread_id: LangGraph thread id (= ``session_id``).
        redis_ckpt: source Redis checkpointer.
        pg_ckpt: destination Postgres checkpointer.

    Returns:
        Number of checkpoints migrated.
    """
    count = await _replay(src=redis_ckpt, dst=pg_ckpt, thread_id=thread_id)
    await redis_ckpt.adelete_thread(thread_id)
    _log.info("memory.migrate_hot_to_cold", thread_id=thread_id, count=count)
    return count


async def migrate_cold_to_hot(
    thread_id: str,
    *,
    pg_ckpt: _AsyncCheckpointer,
    redis_ckpt: RedisCheckpointer,
) -> int:
    """Move a thread from Postgres (cold) back to Redis (hot)."""
    count = await _replay(src=pg_ckpt, dst=redis_ckpt, thread_id=thread_id)
    await pg_ckpt.adelete_thread(thread_id)
    _log.info("memory.migrate_cold_to_hot", thread_id=thread_id, count=count)
    return count
