"""Minimal Redis-backed LangGraph checkpointer with TTL.

Phase-1 stores all checkpoints for a thread in a single Redis hash and
refreshes the TTL on every write. The hot-path working memory IS the
checkpointer; Phase-2 will introduce hot/cold migration where suspended
threads (after ``transfer_to_human``) are moved to a durable Postgres
checkpointer with no TTL.

Limitations
-----------
- ``alist`` returns checkpoints from newest to oldest but does not honor
  the ``before``/``filter`` arguments beyond rough ordering. This is
  sufficient for Phase-1's linear graph; richer history queries can land
  alongside the Postgres checkpointer in Phase-2.
- ``adelete_thread`` removes the entire thread hash atomically.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import redis.asyncio as redis_async
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)


def _thread_id(config: RunnableConfig) -> str:
    cfg = config.get("configurable", {}) if config else {}
    thread_id = cfg.get("thread_id")
    if not thread_id:
        raise ValueError("config['configurable']['thread_id'] is required")
    return str(thread_id)


def _checkpoint_id(config: RunnableConfig) -> str | None:
    cfg = config.get("configurable", {}) if config else {}
    val = cfg.get("checkpoint_id")
    return str(val) if val else None


def _hash_key(thread_id: str) -> str:
    return f"ckpt:{thread_id}"


def _writes_key(thread_id: str, checkpoint_id: str) -> str:
    return f"ckpt_writes:{thread_id}:{checkpoint_id}"


class RedisCheckpointer(BaseCheckpointSaver):
    """Redis-backed checkpointer keyed by ``thread_id`` with shared TTL."""

    def __init__(
        self,
        client: redis_async.Redis,
        *,
        ttl_seconds: int,
    ) -> None:
        super().__init__()
        self._client = client
        self._ttl = ttl_seconds

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any],
    ) -> RunnableConfig:
        thread_id = _thread_id(config)
        ckpt_id = checkpoint["id"]

        ckpt_type, ckpt_blob = self.serde.dumps_typed(checkpoint)
        meta_type, meta_blob = self.serde.dumps_typed(metadata)

        key = _hash_key(thread_id)
        mapping: dict[bytes, bytes] = {
            f"ck:{ckpt_id}:type".encode(): ckpt_type.encode("utf-8"),
            f"ck:{ckpt_id}:blob".encode(): ckpt_blob,
            f"ck:{ckpt_id}:meta_type".encode(): meta_type.encode("utf-8"),
            f"ck:{ckpt_id}:meta_blob".encode(): meta_blob,
            b"latest": ckpt_id.encode("utf-8"),
        }
        parent_id = config.get("configurable", {}).get("checkpoint_id")
        if parent_id:
            mapping[f"ck:{ckpt_id}:parent".encode()] = str(parent_id).encode("utf-8")

        await self._client.hset(key, mapping=mapping)
        await self._client.expire(key, self._ttl)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": ckpt_id,
                "checkpoint_ns": "",
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = _thread_id(config)
        ckpt_id = _checkpoint_id(config)
        if not ckpt_id:
            return
        key = _writes_key(thread_id, ckpt_id)
        mapping: dict[bytes, bytes] = {}
        for idx, (channel, value) in enumerate(writes):
            value_type, value_blob = self.serde.dumps_typed(value)
            mapping[f"{task_id}:{idx}:channel".encode()] = channel.encode("utf-8")
            mapping[f"{task_id}:{idx}:type".encode()] = value_type.encode("utf-8")
            mapping[f"{task_id}:{idx}:blob".encode()] = value_blob
        if mapping:
            await self._client.hset(key, mapping=mapping)
            await self._client.expire(key, self._ttl)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = _thread_id(config)
        ckpt_id = _checkpoint_id(config)
        key = _hash_key(thread_id)

        if ckpt_id is None:
            latest = await self._client.hget(key, b"latest")
            if latest is None:
                return None
            ckpt_id = latest.decode("utf-8")

        return await self._build_tuple(thread_id, ckpt_id)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if not config:
            return
        thread_id = _thread_id(config)
        key = _hash_key(thread_id)
        all_fields = await self._client.hgetall(key)
        ids = sorted({k.decode("utf-8").split(":")[1] for k in all_fields if k.startswith(b"ck:")})
        before_id = _checkpoint_id(before) if before else None
        if before_id:
            ids = [i for i in ids if i < before_id]
        ids.reverse()  # newest first
        if limit is not None:
            ids = ids[:limit]
        for ckpt_id in ids:
            tup = await self._build_tuple(thread_id, ckpt_id)
            if tup is not None:
                yield tup

    async def adelete_thread(self, thread_id: str) -> None:
        key = _hash_key(thread_id)
        keys = [key]
        # Also remove any per-checkpoint write hashes for this thread.
        async for k in self._client.scan_iter(match=f"ckpt_writes:{thread_id}:*"):
            keys.append(k)
        if keys:
            await self._client.delete(*keys)

    async def _build_tuple(
        self,
        thread_id: str,
        checkpoint_id: str,
    ) -> CheckpointTuple | None:
        key = _hash_key(thread_id)
        fields = await self._client.hmget(
            key,
            [
                f"ck:{checkpoint_id}:type".encode(),
                f"ck:{checkpoint_id}:blob".encode(),
                f"ck:{checkpoint_id}:meta_type".encode(),
                f"ck:{checkpoint_id}:meta_blob".encode(),
                f"ck:{checkpoint_id}:parent".encode(),
            ],
        )
        ck_type, ck_blob, mt, mb, parent = fields
        if ck_type is None or ck_blob is None:
            return None

        checkpoint = self.serde.loads_typed((ck_type.decode("utf-8"), ck_blob))
        metadata = (
            self.serde.loads_typed((mt.decode("utf-8"), mb)) if mt and mb else CheckpointMetadata()
        )

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": "",
            }
        }
        parent_config: RunnableConfig | None = None
        if parent is not None:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": parent.decode("utf-8"),
                    "checkpoint_ns": "",
                }
            }
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
        )
