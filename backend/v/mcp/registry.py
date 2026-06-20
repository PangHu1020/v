"""MCP registry: builds the agent's MCP tools via langchain-mcp-adapters.

Replaces the hand-rolled per-server ``MCPClient`` + manual JSON-schema→Pydantic
tool wrapping. ``MultiServerMCPClient`` owns connection/session management and
converts every server's tools to LangChain tools in one flat list
(``get_tools()``), with built-in ``{server}_{tool}`` prefixing for
collision-free names.

Two cross-cutting concerns we keep from the old implementation are layered in
without touching the conversion:

- **Caching + write-opt-out** via :class:`CachingInterceptor` (a
  ``ToolCallInterceptor``): non-write tool results are served from
  :class:`MCPToolCache`; a cache hit short-circuits the handler. Write-class
  tools (per-server ``write_tools``) never cache.
- **OAuth** via :class:`backend.v.mcp.oauth.ClientCredentialsAuth` passed as the
  connection's ``httpx.Auth`` (token fetch + refresh transparent to the adapter).

Note: ``get_tools()`` opens a fresh session per tool call — fine for HTTP,
heavier for stdio (subprocess per call). Acceptable given HTTP is the primary
transport.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from mcp.types import CallToolResult, TextContent

from backend.v.mcp.cache import MCPToolCache
from backend.v.mcp.config import MCPServerConfig
from backend.v.mcp.oauth import ClientCredentialsAuth, ClientCredentialsProvider
from backend.v.utils.logging import bind_request, get_logger

_log = get_logger("mcp.registry")


def _extract_text(result: CallToolResult) -> str:
    """Flatten an MCP ``CallToolResult`` into a single string."""
    if result.isError:
        parts = [c.text for c in result.content if isinstance(c, TextContent)]
        return f"[tool_error] {' '.join(parts) or 'unknown error'}"
    parts = []
    for c in result.content:
        if isinstance(c, TextContent):
            parts.append(c.text)
        else:
            dumper = getattr(c, "model_dump_json", None)
            parts.append(dumper() if callable(dumper) else str(c))
    return "\n".join(parts).strip()


class CachingInterceptor:
    """``ToolCallInterceptor`` that caches non-write tool results.

    Keys on ``(server_name, tool_name, args)`` — ``server_name`` is the
    connection key (our ``MCPServerConfig.id``) and ``tool_name`` is the raw
    (unprefixed) MCP tool name, which is what ``write_tools`` lists. A cache hit
    returns a reconstructed ``CallToolResult`` and skips the handler entirely.
    """

    def __init__(self, cache: MCPToolCache, write_tools_by_server: dict[str, set[str]]) -> None:
        self._cache = cache
        self._write = write_tools_by_server

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
    ) -> MCPToolCallResult:
        server = request.server_name
        tool = request.name
        args = request.args or {}
        is_write = tool in self._write.get(server, set())

        with bind_request(mcp_server=server, mcp_tool=tool):
            if not is_write:
                cached = await self._cache.get(server_id=server, tool_name=tool, arguments=args)
                if cached is not None:
                    _log.debug("mcp.registry.cache_hit", server=server, tool=tool)
                    return CallToolResult(
                        content=[TextContent(type="text", text=str(cached))],
                        isError=False,
                    )

            result = await handler(request)

            # Only cache a successful CallToolResult (handler may also return a
            # ToolMessage / Command — those pass through uncached).
            if not is_write and isinstance(result, CallToolResult) and not result.isError:
                text = _extract_text(result)
                if not text.startswith("[tool_error]"):
                    await self._cache.put(
                        server_id=server, tool_name=tool, arguments=args, value=text
                    )
            return result


class MCPRegistry:
    """Builds + holds the agent's MCP tools via ``MultiServerMCPClient``.

    Keeps the old public surface (``connect_all`` / ``tools`` / ``aclose``) so
    the FastAPI lifespan wiring is unchanged, but the internals are now the
    adapters client + a caching interceptor.
    """

    def __init__(
        self,
        configs: list[MCPServerConfig],
        cache: MCPToolCache,
        *,
        call_timeout_seconds: int,
    ) -> None:
        self._configs = configs
        self._cache = cache
        self._call_timeout = call_timeout_seconds
        self._client: MultiServerMCPClient | None = None
        self._providers: list[ClientCredentialsProvider] = []
        self._tools: list[BaseTool] = []

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    def _build_connections(self) -> dict[str, dict[str, Any]]:
        connections: dict[str, dict[str, Any]] = {}
        for cfg in self._configs:
            auth = None
            if cfg.auth_type == "oauth":
                provider = ClientCredentialsProvider(cfg, request_timeout=float(self._call_timeout))
                self._providers.append(provider)
                auth = ClientCredentialsAuth(provider)
            connections[cfg.id] = cfg.to_connection(auth=auth)
        return connections

    async def connect_all(self) -> None:
        """Build the multi-server client and load all tools (flat list)."""
        if not self._configs:
            return
        write_by_server = {cfg.id: set(cfg.write_tools) for cfg in self._configs if cfg.write_tools}
        interceptor = CachingInterceptor(self._cache, write_by_server)
        self._client = MultiServerMCPClient(
            self._build_connections(),
            tool_interceptors=[interceptor],
            tool_name_prefix=True,
        )
        try:
            self._tools = await self._client.get_tools()
        except Exception as exc:
            _log.error("mcp.registry.get_tools_failed", error=str(exc))
            self._tools = []
            return
        _log.info(
            "mcp.registry.ready",
            servers=len(self._configs),
            tools=len(self._tools),
        )

    async def aclose(self) -> None:
        """No persistent sessions to close (get_tools opens per-call sessions)."""
        self._client = None
        self._providers.clear()


def serialize_tool_for_audit(tool: BaseTool) -> str:
    """Helper used by tests / debugging to confirm a tool's surface."""
    schema = tool.args_schema if isinstance(tool.args_schema, dict) else {}
    if not schema and tool.args_schema is not None:
        getter = getattr(tool.args_schema, "model_json_schema", None)
        schema = getter() if callable(getter) else {}
    return json.dumps(
        {"name": tool.name, "description": tool.description, "schema": schema},
        ensure_ascii=False,
    )
