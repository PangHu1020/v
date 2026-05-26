"""Durable Postgres checkpointer using LangGraph's official AsyncPostgresSaver.

Phase-1 stored all working memory in Redis; that's fine when threads are
short-lived. Phase-2 introduces ``transfer_to_human`` interrupts where a
graph can stay suspended for hours or days waiting for an operator. Redis
30-min TTL is unsafe for those threads, so they migrate to a durable
Postgres-backed checkpointer until the operator resumes.

The official AsyncPostgresSaver uses psycopg3 (separate driver from our
asyncpg app pool) but creates its tables in whichever schema is first on
the connection's ``search_path``. We append ``options=-csearch_path=agent,public``
to the DSN so the LangGraph tables (``checkpoints``, ``checkpoint_blobs``,
``checkpoint_writes``, ``checkpoint_migrations``) land alongside the rest
of the agent schema.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


def _scope_dsn_to_agent_schema(dsn: str) -> str:
    """Append ``search_path=agent,public`` to a PostgreSQL DSN."""
    if "options=" in dsn:
        # Caller already set ``options``; respect it and just hope they did
        # the right thing. Tests cover the unmodified path.
        return dsn
    sep = "&" if "?" in dsn else "?"
    return f"{dsn}{sep}options=-csearch_path%3Dagent,public"


@asynccontextmanager
async def open_pg_checkpointer(dsn: str) -> AsyncIterator[AsyncPostgresSaver]:
    """Open a durable Postgres checkpointer with tables scoped to ``agent``.

    Args:
        dsn: PostgreSQL DSN.

    Yields:
        A ready :class:`AsyncPostgresSaver`. Its tables are created on first
        use via ``setup()``.
    """
    scoped = _scope_dsn_to_agent_schema(dsn)
    async with AsyncPostgresSaver.from_conn_string(scoped) as saver:
        await saver.setup()
        yield saver
