"""Integration test: MCP registry + FastMCP server in-process.

Uses the mcp package's :func:`create_connected_server_and_client_session`
test helper to wire a FastMCP server directly to a client session without
any transport (no stdio/http). The registry's tool-builder + cache
behavior is exercised against a real session.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from backend.v.mcp.cache import MCPToolCache
from backend.v.mcp.client import MCPClient
from backend.v.mcp.config import MCPServerConfig
from backend.v.mcp.registry import MCPRegistry, _extract_text


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


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


async def _build_registry(
    redis: fakeredis.aioredis.FakeRedis,
    *,
    write_tools: list[str] | None = None,
):
    server = _build_server()
    cache = MCPToolCache(redis, l1_ttl_seconds=60, l2_ttl_seconds=3600)
    cfg = MCPServerConfig(
        id="products",
        transport="stdio",
        command="unused",
        write_tools=write_tools or [],
    )
    registry = MCPRegistry([cfg], cache, call_timeout_seconds=10)
    return server, cache, cfg, registry


async def _populate_registry(
    registry: MCPRegistry,
    cfg: MCPServerConfig,
    session,
) -> None:
    """Inject a pre-built ClientSession into a fresh MCPClient and populate
    the registry's tool list. Bypasses :meth:`MCPClient.connect` since the
    in-memory test helper already initialized the session for us."""
    client = MCPClient(cfg, call_timeout_seconds=10)
    client._session = session
    registry._clients[cfg.id] = client
    listing = await client.list_tools()
    for mcp_tool in listing.tools:
        lc_tool = registry._build_langchain_tool(client, mcp_tool)
        registry._tools.append(lc_tool)


class TestRegistryAgainstFastMCP:
    async def test_lists_and_qualifies_tools(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        server, _, cfg, registry = await _build_registry(redis_client)
        async with create_connected_server_and_client_session(server) as session:
            await _populate_registry(registry, cfg, session)
            names = sorted(t.name for t in registry.tools)
            assert names == [
                "products__get_product",
                "products__place_order",
                "products__search",
            ]

    async def test_call_returns_text(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        server, _, cfg, registry = await _build_registry(redis_client)
        async with create_connected_server_and_client_session(server) as session:
            await _populate_registry(registry, cfg, session)
            tool = next(t for t in registry.tools if t.name == "products__get_product")
            out = await tool.ainvoke({"product_id": "P001"})
            assert "iPhone-P001" in out
            assert "stock=42" in out

    async def test_read_tool_is_cached(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        server, cache, cfg, registry = await _build_registry(redis_client)
        async with create_connected_server_and_client_session(server) as session:
            await _populate_registry(registry, cfg, session)
            tool = next(t for t in registry.tools if t.name == "products__search")

            # First call: hits the server.
            await tool.ainvoke({"query": "iphone", "limit": 5})
            cached = await cache.get(
                server_id="products",
                tool_name="search",
                arguments={"query": "iphone", "limit": 5},
            )
            assert cached is not None
            assert "iphone" in cached

    async def test_write_tool_is_not_cached(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        server, cache, cfg, registry = await _build_registry(
            redis_client, write_tools=["place_order"]
        )
        async with create_connected_server_and_client_session(server) as session:
            await _populate_registry(registry, cfg, session)
            tool = next(t for t in registry.tools if t.name == "products__place_order")
            out = await tool.ainvoke({"product_id": "P001", "quantity": 2})
            assert "ordered 2xP001" in out
            cached = await cache.get(
                server_id="products",
                tool_name="place_order",
                arguments={"product_id": "P001", "quantity": 2},
            )
            assert cached is None

    async def test_args_schema_built_from_input_schema(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        server, _, cfg, registry = await _build_registry(redis_client)
        async with create_connected_server_and_client_session(server) as session:
            await _populate_registry(registry, cfg, session)
            tool = next(t for t in registry.tools if t.name == "products__search")
            schema = tool.args_schema.model_json_schema()
            props = schema["properties"]
            assert "query" in props
            assert "limit" in props
            # 'query' is required.
            assert "query" in schema.get("required", [])


class TestExtractTextEdgeCases:
    def test_empty_content(self) -> None:
        from mcp.types import CallToolResult

        result = CallToolResult(content=[], isError=False)
        assert _extract_text(result) == ""

    def test_error_payload_marked(self) -> None:
        from mcp.types import CallToolResult, TextContent

        result = CallToolResult(
            content=[TextContent(type="text", text="not found")],
            isError=True,
        )
        out = _extract_text(result)
        assert out.startswith("[tool_error]")
        assert "not found" in out
