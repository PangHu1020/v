"""Model Context Protocol client.

Connects to one or more MCP servers (stdio or HTTP/SSE) via
``langchain-mcp-adapters``, exposes their tools to the agent as one flat list,
and caches non-write-class results.

The server *list* is loaded from ``.agent/config.json`` by
:func:`backend.v.configs.agent_config.load_agent_config` at app startup.

Layout:

- :mod:`config`   — ``MCPServerConfig`` schema + connection mapping.
- :mod:`oauth`    — ``ClientCredentialsProvider`` M2M token provider +
  ``ClientCredentialsAuth`` (``httpx.Auth`` adapter for the connections).
- :mod:`cache`    — L1 in-memory + L2 Redis cache for tool results.
- :mod:`registry` — builds tools via ``MultiServerMCPClient`` + a
  ``CachingInterceptor``; hands back LangChain tools for the agent.
"""

from backend.v.mcp.cache import MCPToolCache
from backend.v.mcp.config import MCPServerConfig
from backend.v.mcp.oauth import ClientCredentialsAuth, ClientCredentialsProvider, OAuthError
from backend.v.mcp.registry import CachingInterceptor, MCPRegistry

__all__ = [
    "CachingInterceptor",
    "ClientCredentialsAuth",
    "ClientCredentialsProvider",
    "MCPRegistry",
    "MCPServerConfig",
    "MCPToolCache",
    "OAuthError",
]
