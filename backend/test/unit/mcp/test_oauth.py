"""Unit tests for the OAuth client_credentials token provider."""

from __future__ import annotations

import httpx
import pytest
import respx

from backend.v.mcp.config import MCPServerConfig
from backend.v.mcp.oauth import ClientCredentialsProvider, OAuthError

_TOKEN_URL = "http://mock/token"


def _cfg() -> MCPServerConfig:
    return MCPServerConfig(
        id="shop",
        transport="http",
        url="http://mock/mcp",
        auth_type="oauth",
        oauth_token_url=_TOKEN_URL,
        oauth_client_id="agent",
        oauth_client_secret="secret",
        oauth_scope="tools:call",
    )


class TestTokenFetch:
    @respx.mock
    async def test_fetches_and_returns_bearer(self) -> None:
        route = respx.post(_TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "abc", "expires_in": 3600})
        )
        p = ClientCredentialsProvider(_cfg())
        assert await p.token() == "abc"
        assert route.called
        headers = await p.auth_headers()
        assert headers == {"Authorization": "Bearer abc"}

    @respx.mock
    async def test_caches_token_across_calls(self) -> None:
        route = respx.post(_TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "abc", "expires_in": 3600})
        )
        p = ClientCredentialsProvider(_cfg())
        await p.token()
        await p.token()
        await p.token()
        assert route.call_count == 1  # cached, only one network hit

    @respx.mock
    async def test_refetches_when_expired(self) -> None:
        respx.post(_TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "t1", "expires_in": 3600})
        )
        p = ClientCredentialsProvider(_cfg())
        assert await p.token() == "t1"
        # Force expiry, swap the response, confirm a re-fetch.
        p._expires_at = 0.0
        respx.post(_TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "t2", "expires_in": 3600})
        )
        assert await p.token() == "t2"

    @respx.mock
    async def test_includes_scope_in_request(self) -> None:
        captured = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content.decode()
            return httpx.Response(200, json={"access_token": "abc", "expires_in": 60})

        respx.post(_TOKEN_URL).mock(side_effect=_handler)
        await ClientCredentialsProvider(_cfg()).token()
        assert "grant_type=client_credentials" in captured["body"]
        assert "scope=tools" in captured["body"]


class TestTokenErrors:
    @respx.mock
    async def test_non_200_raises(self) -> None:
        respx.post(_TOKEN_URL).mock(return_value=httpx.Response(401, text="bad client"))
        with pytest.raises(OAuthError, match="returned 401"):
            await ClientCredentialsProvider(_cfg()).token()

    @respx.mock
    async def test_missing_access_token_raises(self) -> None:
        respx.post(_TOKEN_URL).mock(return_value=httpx.Response(200, json={"token_type": "Bearer"}))
        with pytest.raises(OAuthError, match="missing access_token"):
            await ClientCredentialsProvider(_cfg()).token()

    @respx.mock
    async def test_network_error_raises(self) -> None:
        respx.post(_TOKEN_URL).mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(OAuthError, match="token request failed"):
            await ClientCredentialsProvider(_cfg()).token()
