"""Model Context Protocol client (Phase-2 P1).

Connects to one or more MCP servers (stdio or HTTP/SSE), exposes their
tools to the agent, and caches non-write-class results.

Layout:

- :mod:`config`   — ``MCPServerConfig`` schema + JSON parser.
- :mod:`client`   — ``MCPClient`` long-lived connection wrapper.
- :mod:`cache`    — L1 in-memory + L2 Redis cache for tool results.
- :mod:`registry` — load servers from settings, manage their lifecycle,
  hand back LangChain ``StructuredTool`` instances for the agent.
"""

from backend.v.mcp.cache import MCPToolCache
from backend.v.mcp.client import MCPClient, MCPClientError
from backend.v.mcp.config import MCPServerConfig, parse_servers
from backend.v.mcp.registry import MCPRegistry

__all__ = [
    "MCPClient",
    "MCPClientError",
    "MCPRegistry",
    "MCPServerConfig",
    "MCPToolCache",
    "parse_servers",
]
