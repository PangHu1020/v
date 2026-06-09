"""MCP client: long-lived connection wrapping :class:`mcp.ClientSession`.

Supports stdio, streamable HTTP, and SSE transports. For HTTP-based
transports, auth headers are resolved at connect time:

- ``api_key``: static Bearer header from config.
- ``oauth``: client_credentials bearer fetched via
  :class:`backend.v.mcp.oauth.ClientCredentialsProvider` (cached + refreshed).
- stdio servers authenticate via env vars / the host filesystem.

Concurrency: a single :class:`MCPClient` corresponds to a single MCP
session. Multiple agent turns concurrently calling ``call_tool`` is safe
because the underlying ClientSession serializes JSON-RPC requests.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, ListToolsResult

from backend.v.mcp.config import MCPServerConfig
from backend.v.mcp.oauth import ClientCredentialsProvider
from backend.v.utils.logging import get_logger

_log = get_logger("mcp.client")


class MCPClientError(Exception):
    """Raised on connection or tool-call failures."""


class MCPClient:
    """One MCP server connection + ClientSession."""

    def __init__(self, config: MCPServerConfig, *, call_timeout_seconds: int = 30) -> None:
        self._config = config
        self._call_timeout = timedelta(seconds=call_timeout_seconds)
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._oauth: ClientCredentialsProvider | None = (
            ClientCredentialsProvider(config) if config.auth_type == "oauth" else None
        )

    @property
    def config(self) -> MCPServerConfig:
        return self._config

    async def _resolve_headers(self) -> dict[str, str]:
        """Static headers plus a freshly-resolved OAuth bearer if configured."""
        headers = self._config.static_headers()
        if self._oauth is not None:
            headers.update(await self._oauth.auth_headers())
        return headers

    async def connect(self) -> None:
        """Open the transport, build the session, and call ``initialize``.

        Idempotent on already-connected clients.
        """
        if self._session is not None:
            return
        stack = AsyncExitStack()
        try:
            read, write = await self._open_transport(stack)
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        self._session = session
        _log.info(
            "mcp.client.connected",
            server_id=self._config.id,
            transport=self._config.transport,
            auth=self._config.auth_type,
        )

    async def aclose(self) -> None:
        """Close session + transport. Safe to call multiple times."""
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except BaseException:
                # MCP's anyio task groups occasionally surface cancellation
                # noise on close. We log + swallow so shutdown finishes.
                _log.warning("mcp.client.close_noise", server_id=self._config.id)
        self._stack = None
        self._session = None
        _log.info("mcp.client.closed", server_id=self._config.id)

    async def list_tools(self) -> ListToolsResult:
        if self._session is None:
            raise MCPClientError(f"server {self._config.id!r} is not connected")
        return await self._session.list_tools()

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        if self._session is None:
            raise MCPClientError(f"server {self._config.id!r} is not connected")
        try:
            return await asyncio.wait_for(
                self._session.call_tool(name, arguments or {}),
                timeout=self._call_timeout.total_seconds(),
            )
        except TimeoutError as exc:
            raise MCPClientError(f"server {self._config.id!r} tool {name!r} timed out") from exc

    async def _open_transport(self, stack: AsyncExitStack) -> tuple[Any, Any]:
        cfg = self._config
        if cfg.transport == "stdio":
            params = StdioServerParameters(
                command=cfg.command,
                args=list(cfg.args),
                env=dict(cfg.env) if cfg.env else None,
            )
            ctx = stdio_client(params)
            transport = await stack.enter_async_context(ctx)
            return transport[0], transport[1]
        headers = await self._resolve_headers()
        if cfg.transport == "http":
            ctx = streamablehttp_client(cfg.url, headers=headers or None)
            transport = await stack.enter_async_context(ctx)
            return transport[0], transport[1]
        if cfg.transport == "sse":
            ctx = sse_client(cfg.url, headers=headers or None)
            transport = await stack.enter_async_context(ctx)
            return transport[0], transport[1]
        raise MCPClientError(f"unsupported transport: {cfg.transport}")
