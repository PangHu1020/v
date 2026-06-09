"""OAuth 2.0 client_credentials token provider for MCP HTTP transports.

Machine-to-machine auth: the agent backend exchanges a ``client_id`` +
``client_secret`` for a short-lived bearer token at the server's token
endpoint, caches it, and refreshes shortly before expiry. No user
interaction, no browser redirect.

One :class:`ClientCredentialsProvider` per MCP server config. The token is
cached in memory and protected by an ``asyncio.Lock`` so concurrent agent
turns don't trigger a stampede of token requests.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from backend.v.mcp.config import MCPServerConfig
from backend.v.utils.logging import get_logger

_log = get_logger("mcp.oauth")

# Refresh this many seconds BEFORE the token actually expires, to avoid
# racing a request against expiry.
_REFRESH_SKEW_SECONDS = 30
# Fallback lifetime when the token endpoint omits ``expires_in``.
_DEFAULT_EXPIRES_IN = 3600


class OAuthError(Exception):
    """Raised when a token cannot be obtained."""


class ClientCredentialsProvider:
    """Fetch + cache an OAuth client_credentials bearer token."""

    def __init__(self, config: MCPServerConfig, *, request_timeout: float = 10.0) -> None:
        self._config = config
        self._timeout = request_timeout
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    def _is_fresh(self) -> bool:
        return self._token is not None and time.monotonic() < self._expires_at

    async def token(self) -> str:
        """Return a valid bearer token, fetching/refreshing if needed."""
        if self._is_fresh():
            return self._token  # type: ignore[return-value]
        async with self._lock:
            # Re-check inside the lock — another coroutine may have refreshed.
            if self._is_fresh():
                return self._token  # type: ignore[return-value]
            await self._fetch()
            return self._token  # type: ignore[return-value]

    async def _fetch(self) -> None:
        cfg = self._config
        data = {
            "grant_type": "client_credentials",
            "client_id": cfg.oauth_client_id,
            "client_secret": cfg.oauth_client_secret,
        }
        if cfg.oauth_scope:
            data["scope"] = cfg.oauth_scope
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as http:
                resp = await http.post(cfg.oauth_token_url, data=data)
        except httpx.HTTPError as exc:
            raise OAuthError(f"server {cfg.id!r}: token request failed: {exc}") from exc
        if resp.status_code != 200:
            raise OAuthError(
                f"server {cfg.id!r}: token endpoint returned {resp.status_code}: {resp.text[:200]}"
            )
        payload = resp.json()
        token = payload.get("access_token")
        if not token:
            raise OAuthError(f"server {cfg.id!r}: token response missing access_token")
        expires_in = int(payload.get("expires_in", _DEFAULT_EXPIRES_IN))
        self._token = token
        self._expires_at = time.monotonic() + max(expires_in - _REFRESH_SKEW_SECONDS, 1)
        _log.info("mcp.oauth.token_refreshed", server_id=cfg.id, expires_in=expires_in)

    async def auth_headers(self) -> dict[str, str]:
        """Return ``{"Authorization": "Bearer <token>"}`` for the current token."""
        return {"Authorization": f"Bearer {await self.token()}"}
