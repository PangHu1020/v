"""MCP registry tests: caching interceptor (unit) + adapter conversion (FastMCP).

The interceptor logic is tested in isolation with a fake handler. The full
adapter path (MCP tool → LangChain tool, with the interceptor composed in) is
tested against an in-process FastMCP server via ``load_mcp_tools(session, ...)``
— the same conversion ``MultiServerMCPClient.get_tools()`` performs, but driven
by the in-memory test session so no transport is needed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent

from backend.v.mcp.cache import MCPToolCache
from backend.v.mcp.registry import CachingInterceptor, _extract_text


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


def _ok_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=False)


def _req(server: str, name: str, args: dict) -> MCPToolCallRequest:
    return MCPToolCallRequest(name=name, args=args, server_name=server, headers=None, runtime=None)


# ── CachingInterceptor in isolation ───────────────────────────────────────────


class TestCachingInterceptor:
    async def test_caches_read_and_short_circuits(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        cache = MCPToolCache(redis_client, l1_ttl_seconds=60, l2_ttl_seconds=3600)
        ic = CachingInterceptor(cache, write_tools_by_server={})
        handler = AsyncMock(return_value=_ok_result("hit-42"))

        req = _req("products", "search", {"q": "iphone"})
        out1 = await ic(req, handler)
        out2 = await ic(req, handler)

        assert _extract_text(out1) == "hit-42"
        assert _extract_text(out2) == "hit-42"
        # Second call served from cache → handler only invoked once.
        handler.assert_awaited_once()

    async def test_write_tool_not_cached(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        cache = MCPToolCache(redis_client, l1_ttl_seconds=60, l2_ttl_seconds=3600)
        ic = CachingInterceptor(cache, write_tools_by_server={"orders": {"place"}})
        handler = AsyncMock(return_value=_ok_result("ordered"))

        req = _req("orders", "place", {"sku": "P1"})
        await ic(req, handler)
        await ic(req, handler)

        # Write tool → never cached → handler invoked both times.
        assert handler.await_count == 2
        assert (
            await cache.get(server_id="orders", tool_name="place", arguments={"sku": "P1"}) is None
        )

    async def test_error_result_not_cached(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        cache = MCPToolCache(redis_client, l1_ttl_seconds=60, l2_ttl_seconds=3600)
        ic = CachingInterceptor(cache, write_tools_by_server={})
        err = CallToolResult(content=[TextContent(type="text", text="boom")], isError=True)
        handler = AsyncMock(return_value=err)

        req = _req("svc", "flaky", {})
        await ic(req, handler)
        await ic(req, handler)

        # Errors must not poison the cache → handler invoked both times.
        assert handler.await_count == 2


# ── Full adapter conversion + interceptor against FastMCP ─────────────────────


def _build_server() -> FastMCP:
    server = FastMCP("test-products")

    @server.tool()
    def get_product(product_id: str) -> str:
        """Return a product description by its id."""
        return f"name=iPhone-{product_id} stock=42"

    @server.tool()
    def search(query: str, limit: int = 10) -> str:
        """Search products by query."""
        return f"results for '{query}' (limit={limit})"

    @server.tool()
    def place_order(product_id: str, quantity: int) -> str:
        """Place an order. Side effect: changes inventory."""
        return f"ordered {quantity}x{product_id}"

    return server


async def _load_tools(session, cache, *, write_tools: set[str] | None = None):
    ic = CachingInterceptor(cache, write_tools_by_server={"products": write_tools or set()})
    return await load_mcp_tools(
        session,
        server_name="products",
        tool_name_prefix=True,
        tool_interceptors=[ic],
    )


class TestAdapterConversion:
    async def test_lists_and_prefixes_tools(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        cache = MCPToolCache(redis_client, l1_ttl_seconds=60, l2_ttl_seconds=3600)
        async with create_connected_server_and_client_session(_build_server()) as session:
            tools = await _load_tools(session, cache)
            names = sorted(t.name for t in tools)
            assert names == [
                "products_get_product",
                "products_place_order",
                "products_search",
            ]

    async def test_call_returns_text(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        cache = MCPToolCache(redis_client, l1_ttl_seconds=60, l2_ttl_seconds=3600)
        async with create_connected_server_and_client_session(_build_server()) as session:
            tools = await _load_tools(session, cache)
            tool = next(t for t in tools if t.name == "products_get_product")
            out = await tool.ainvoke({"product_id": "P001"})
            assert "iPhone-P001" in str(out)

    async def test_read_tool_is_cached(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        cache = MCPToolCache(redis_client, l1_ttl_seconds=60, l2_ttl_seconds=3600)
        async with create_connected_server_and_client_session(_build_server()) as session:
            tools = await _load_tools(session, cache)
            tool = next(t for t in tools if t.name == "products_search")
            await tool.ainvoke({"query": "iphone", "limit": 5})
            cached = await cache.get(
                server_id="products",
                tool_name="search",
                arguments={"query": "iphone", "limit": 5},
            )
            assert cached is not None
            assert "iphone" in cached

    async def test_write_tool_is_not_cached(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        cache = MCPToolCache(redis_client, l1_ttl_seconds=60, l2_ttl_seconds=3600)
        async with create_connected_server_and_client_session(_build_server()) as session:
            tools = await _load_tools(session, cache, write_tools={"place_order"})
            tool = next(t for t in tools if t.name == "products_place_order")
            out = await tool.ainvoke({"product_id": "P001", "quantity": 2})
            assert "ordered 2xP001" in str(out)
            cached = await cache.get(
                server_id="products",
                tool_name="place_order",
                arguments={"product_id": "P001", "quantity": 2},
            )
            assert cached is None


class TestExtractText:
    def test_empty_content(self) -> None:
        assert _extract_text(CallToolResult(content=[], isError=False)) == ""

    def test_error_payload_marked(self) -> None:
        result = CallToolResult(content=[TextContent(type="text", text="not found")], isError=True)
        out = _extract_text(result)
        assert out.startswith("[tool_error]")
        assert "not found" in out


# ── Registry wiring: connection building + lifecycle ──────────────────────────


class TestRegistryWiring:
    async def test_empty_configs_no_op(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        from backend.v.mcp.registry import MCPRegistry

        cache = MCPToolCache(redis_client, l1_ttl_seconds=60, l2_ttl_seconds=3600)
        reg = MCPRegistry([], cache, call_timeout_seconds=10)
        await reg.connect_all()
        assert reg.tools == []
        await reg.aclose()  # safe no-op

    def test_build_connections_api_key_and_oauth(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        from backend.v.mcp.config import MCPServerConfig
        from backend.v.mcp.oauth import ClientCredentialsAuth
        from backend.v.mcp.registry import MCPRegistry

        cache = MCPToolCache(redis_client, l1_ttl_seconds=60, l2_ttl_seconds=3600)
        api_cfg = MCPServerConfig(
            id="shop", transport="http", url="http://x/mcp", auth_type="api_key", api_key="k"
        )
        oauth_cfg = MCPServerConfig(
            id="orders",
            transport="sse",
            url="http://y/sse",
            auth_type="oauth",
            oauth_token_url="http://y/token",
            oauth_client_id="a",
            oauth_client_secret="s",
        )
        reg = MCPRegistry([api_cfg, oauth_cfg], cache, call_timeout_seconds=10)
        conns = reg._build_connections()

        # api_key → static bearer header, no httpx auth object
        assert conns["shop"]["headers"]["Authorization"] == "Bearer k"
        assert "auth" not in conns["shop"]
        assert conns["shop"]["transport"] == "streamable_http"

        # oauth → ClientCredentialsAuth passed as connection auth
        assert isinstance(conns["orders"]["auth"], ClientCredentialsAuth)
        assert conns["orders"]["transport"] == "sse"
        assert len(reg._providers) == 1  # provider tracked for lifecycle

    async def test_get_tools_failure_degrades_to_empty(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        from unittest.mock import patch

        from backend.v.mcp.config import MCPServerConfig
        from backend.v.mcp.registry import MCPRegistry

        cache = MCPToolCache(redis_client, l1_ttl_seconds=60, l2_ttl_seconds=3600)
        cfg = MCPServerConfig(id="svc", transport="http", url="http://unreachable/mcp")
        reg = MCPRegistry([cfg], cache, call_timeout_seconds=10)

        async def _boom(self):
            raise ConnectionError("server down")

        with patch("langchain_mcp_adapters.client.MultiServerMCPClient.get_tools", _boom):
            await reg.connect_all()
        # Failure logged, degrades to empty tool list (doesn't crash startup).
        assert reg.tools == []
