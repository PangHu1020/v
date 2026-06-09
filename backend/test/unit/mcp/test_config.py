"""Unit tests for ``backend.v.mcp.config``."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from backend.v.mcp.config import MCPServerConfig, parse_servers


class TestMCPServerConfigValidation:
    def test_stdio_minimal(self) -> None:
        cfg = MCPServerConfig(
            id="products", transport="stdio", command="python", args=["-m", "products"]
        )
        assert cfg.id == "products"
        assert cfg.command == "python"
        assert cfg.args == ["-m", "products"]
        assert cfg.auth_type == "none"

    def test_stdio_requires_command(self) -> None:
        with pytest.raises(ValidationError, match="command"):
            MCPServerConfig(id="x", transport="stdio")

    def test_http_requires_url(self) -> None:
        with pytest.raises(ValidationError, match="url"):
            MCPServerConfig(id="x", transport="http")

    def test_sse_requires_url(self) -> None:
        with pytest.raises(ValidationError, match="url"):
            MCPServerConfig(id="x", transport="sse")

    def test_api_key_auth_requires_key(self) -> None:
        with pytest.raises(ValidationError, match="api_key"):
            MCPServerConfig(
                id="x",
                transport="http",
                url="https://x.example/mcp",
                auth_type="api_key",
            )

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MCPServerConfig(id="", transport="stdio", command="x")


class TestAuthHeaders:
    def test_no_auth_returns_extras_only(self) -> None:
        cfg = MCPServerConfig(
            id="x",
            transport="http",
            url="https://x/",
            extra_headers={"X-Custom": "v"},
        )
        assert cfg.static_headers() == {"X-Custom": "v"}

    def test_api_key_adds_bearer(self) -> None:
        cfg = MCPServerConfig(
            id="x",
            transport="http",
            url="https://x/",
            auth_type="api_key",
            api_key="sk-test",
        )
        assert cfg.static_headers() == {"Authorization": "Bearer sk-test"}

    def test_api_key_merges_with_extras(self) -> None:
        cfg = MCPServerConfig(
            id="x",
            transport="http",
            url="https://x/",
            auth_type="api_key",
            api_key="sk",
            extra_headers={"X-Trace": "1"},
        )
        h = cfg.static_headers()
        assert h["Authorization"] == "Bearer sk"
        assert h["X-Trace"] == "1"


class TestParseServers:
    def test_empty_string_yields_empty_list(self) -> None:
        assert parse_servers("") == []

    def test_empty_array(self) -> None:
        assert parse_servers("[]") == []

    def test_single_stdio(self) -> None:
        servers = parse_servers(
            json.dumps(
                [
                    {
                        "id": "products",
                        "transport": "stdio",
                        "command": "python",
                        "args": ["-m", "products_mcp"],
                    }
                ]
            )
        )
        assert len(servers) == 1
        assert servers[0].id == "products"

    def test_two_servers(self) -> None:
        servers = parse_servers(
            json.dumps(
                [
                    {"id": "a", "transport": "stdio", "command": "x"},
                    {
                        "id": "b",
                        "transport": "http",
                        "url": "https://b/",
                        "auth_type": "api_key",
                        "api_key": "k",
                    },
                ]
            )
        )
        assert [s.id for s in servers] == ["a", "b"]

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_servers("not-json")

    def test_non_array_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a JSON array"):
            parse_servers('{"id": "x"}')

    def test_invalid_entry_raises(self) -> None:
        with pytest.raises(ValidationError):
            parse_servers(json.dumps([{"id": "no-transport"}]))


class TestWriteToolsOptOut:
    def test_recorded(self) -> None:
        cfg = MCPServerConfig(
            id="orders",
            transport="stdio",
            command="x",
            write_tools=["place_order", "cancel_order"],
        )
        assert "place_order" in cfg.write_tools
        assert "cancel_order" in cfg.write_tools


class TestOAuthConfig:
    def test_oauth_requires_token_url_and_creds(self) -> None:
        with pytest.raises(ValidationError, match="auth_type=oauth requires"):
            MCPServerConfig(
                id="shop", transport="http", url="http://x/mcp", auth_type="oauth"
            )

    def test_oauth_valid(self) -> None:
        cfg = MCPServerConfig(
            id="shop",
            transport="http",
            url="http://x/mcp",
            auth_type="oauth",
            oauth_token_url="http://x/token",
            oauth_client_id="agent",
            oauth_client_secret="secret",
            oauth_scope="tools:call",
        )
        assert cfg.auth_type == "oauth"
        # OAuth bearer is NOT a static header — fetched dynamically by the client.
        assert "Authorization" not in cfg.static_headers()
