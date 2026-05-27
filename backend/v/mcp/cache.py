"""Two-tier cache for MCP tool results.

L1: per-process in-memory dict, default 5-min TTL. Avoids the network
hop for hot tool calls within a single agent turn or across rapid turns.

L2: Redis, default 24-hour TTL. Survives restarts and is shared across
worker processes; backstops L1 misses.

Cache key: ``mcp:cache:{server_id}:{tool_name}:{sha256(args)[:16]}``.
Args are canonicalized via ``json.dumps(..., sort_keys=True)`` so logically
equivalent argument dicts collide.

Write-class tools (those with side effects: place_order, refund, etc.)
opt out of caching at the registry layer; this module never knows about
that policy — it just stores and returns whatever it's told.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import orjson
import redis.asyncio as redis_async

from backend.v.utils.logging import get_logger

_log = get_logger("mcp.cache")


def _cache_key(server_id: str, tool_name: str, arguments: dict[str, Any]) -> str:
    canonical = json.dumps(arguments, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:16]
    return f"mcp:cache:{server_id}:{tool_name}:{digest}"


class MCPToolCache:
    """L1 in-memory + L2 Redis cache for MCP tool results."""

    def __init__(
        self,
        redis: redis_async.Redis,
        *,
        l1_ttl_seconds: int,
        l2_ttl_seconds: int,
    ) -> None:
        self._redis = redis
        self._l1_ttl = l1_ttl_seconds
        self._l2_ttl = l2_ttl_seconds
        # key -> (expires_at_monotonic, value)
        self._l1: dict[str, tuple[float, Any]] = {}

    def _l1_get(self, key: str) -> Any | None:
        entry = self._l1.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            del self._l1[key]
            return None
        return value

    def _l1_put(self, key: str, value: Any) -> None:
        self._l1[key] = (time.monotonic() + self._l1_ttl, value)

    async def get(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any | None:
        """Return the cached result if hot in either tier, else ``None``."""
        key = _cache_key(server_id, tool_name, arguments)

        l1_hit = self._l1_get(key)
        if l1_hit is not None:
            _log.debug("mcp.cache.l1_hit", key=key)
            return l1_hit

        raw = await self._redis.get(key)
        if raw is None:
            _log.debug("mcp.cache.miss", key=key)
            return None
        try:
            value = orjson.loads(raw)
        except orjson.JSONDecodeError:
            _log.warning("mcp.cache.l2_corrupt", key=key)
            await self._redis.delete(key)
            return None
        # Refill L1 on L2 hit.
        self._l1_put(key, value)
        _log.debug("mcp.cache.l2_hit", key=key)
        return value

    async def put(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        value: Any,
    ) -> None:
        """Store ``value`` in both tiers."""
        key = _cache_key(server_id, tool_name, arguments)
        self._l1_put(key, value)
        await self._redis.set(key, orjson.dumps(value), ex=self._l2_ttl)

    def clear_l1(self) -> None:
        """Purge the in-memory tier. Used in tests; production shouldn't need it."""
        self._l1.clear()

    async def invalidate(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Drop a single entry from both tiers."""
        key = _cache_key(server_id, tool_name, arguments)
        self._l1.pop(key, None)
        await self._redis.delete(key)
