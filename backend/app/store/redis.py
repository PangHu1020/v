"""Async Redis client factory.

Used as the shared backing for:

- the bus (Redis Streams + consumer groups, sharded by ``(channel, channel_user_id)``),
- working memory (= LangGraph checkpointer) with 30-min TTL,
- the channel-layer 500ms debounce.

A single client is created in the FastAPI lifespan and injected via
``Depends`` (or app context) into routers, the bus producer/consumer, and
``/v/`` modules.
"""

from __future__ import annotations

import redis.asyncio as redis_async


async def create_client(url: str) -> redis_async.Redis:
    """Create an async Redis client from the URL.

    The connection pool is lazily established on first use. ``decode_responses``
    is left at the default ``False`` because Streams entries and binary protobuf
    payloads round-trip more cleanly as bytes; callers that want strings
    decode locally.

    Args:
        url: ``redis://host:port/db`` style URL.

    Returns:
        A ready ``redis.asyncio.Redis`` client.
    """
    client = redis_async.from_url(url, decode_responses=False)
    return client


async def close_client(client: redis_async.Redis) -> None:
    """Close the Redis client and its underlying connection pool."""
    await client.aclose()


async def redis_health(client: redis_async.Redis) -> bool:
    """Return ``True`` iff the Redis ``PING`` round-trip succeeds."""
    try:
        return bool(await client.ping())
    except Exception:
        return False
