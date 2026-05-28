"""Public Protocol describing the subset of LangGraph checkpointer
behavior the rest of the codebase relies on.

Lets workers and the migration helpers depend on a stable interface
instead of a concrete class. Both :class:`RedisCheckpointer` and
``langgraph_checkpoint_postgres.AsyncPostgresSaver`` satisfy this
Protocol via duck typing, which is enough for type-checkers and for
swapping implementations in tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol


class AsyncCheckpointer(Protocol):
    """Subset of ``BaseCheckpointSaver`` we actually depend on."""

    async def aput(
        self,
        config: Any,
        checkpoint: Any,
        metadata: Any,
        new_versions: Any,
    ) -> Any: ...

    async def aput_writes(
        self,
        config: Any,
        writes: Any,
        task_id: str,
        task_path: str = "",
    ) -> None: ...

    async def aget_tuple(self, config: Any) -> Any | None: ...

    def alist(
        self,
        config: Any,
        *,
        filter: dict[str, Any] | None = None,
        before: Any | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[Any]: ...

    async def adelete_thread(self, thread_id: str) -> None: ...
