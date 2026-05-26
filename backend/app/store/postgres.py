"""PostgreSQL pool helpers built on ``asyncpg``.

The pool registers two type codecs on every connection:

- ``jsonb``: Python ``dict``/``list`` <-> Postgres jsonb via ``json.dumps`` /
  ``json.loads``. Without this, asyncpg returns jsonb as a raw string.
- ``vector(N)``: pgvector codec from the ``pgvector`` Python package, so that
  ``agent.memory_episodes.embedding`` can be passed and read as a Python list
  of floats.

``acquire_with_schema`` wraps ``Pool.acquire()`` and issues
``SET LOCAL search_path = <schema>, public`` so caller code can write
unqualified table references inside one of our three schemas.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

import asyncpg
from pgvector.asyncpg import register_vector

SchemaName = Literal["agent", "dw", "meta"]


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register codecs on every new connection added to the pool."""
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    await register_vector(conn)


async def create_pool(
    dsn: str,
    *,
    min_size: int = 1,
    max_size: int = 10,
) -> asyncpg.Pool:
    """Create an ``asyncpg`` pool with codecs registered for jsonb + vector.

    Args:
        dsn: PostgreSQL connection string.
        min_size: Minimum number of pooled connections.
        max_size: Maximum number of pooled connections.

    Returns:
        A ready ``asyncpg.Pool``.
    """
    pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=min_size,
        max_size=max_size,
        init=_init_connection,
    )
    if pool is None:
        raise RuntimeError("asyncpg.create_pool returned None")
    return pool


async def close_pool(pool: asyncpg.Pool) -> None:
    """Close the pool gracefully, terminating in-flight connections after a grace period."""
    await pool.close()


@asynccontextmanager
async def acquire_with_schema(
    pool: asyncpg.Pool,
    schema: SchemaName,
) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a pooled connection scoped to one of the project schemas.

    Issues ``SET LOCAL search_path = <schema>, public`` inside an implicit
    transaction so the override is automatically reverted on connection
    return.

    Args:
        pool: Pool returned by :func:`create_pool`.
        schema: One of ``agent``, ``dw``, ``meta``.

    Yields:
        A connection with ``search_path`` set for the duration of the block.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL search_path = {schema}, public")
            yield conn


async def pg_health(pool: asyncpg.Pool) -> bool:
    """Return ``True`` iff a trivial round-trip query succeeds."""
    try:
        async with pool.acquire() as conn:
            value = await conn.fetchval("SELECT 1")
        return value == 1
    except Exception:
        return False
