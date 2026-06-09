"""MCP server configuration model + JSON parser."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Transport = Literal["stdio", "http", "sse"]
AuthType = Literal["none", "api_key", "oauth"]


class MCPServerConfig(BaseModel):
    """Connection parameters for a single MCP server."""

    id: str = Field(min_length=1, description="Stable identifier; namespaces tool names.")
    transport: Transport
    description: str = ""

    # stdio transport
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    # http / sse transport
    url: str = ""
    auth_type: AuthType = "none"
    api_key: str = ""
    extra_headers: dict[str, str] = Field(default_factory=dict)

    # OAuth 2.0 client_credentials (machine-to-machine) — used when auth_type="oauth".
    oauth_token_url: str = ""
    oauth_client_id: str = ""
    oauth_client_secret: str = ""
    oauth_scope: str = ""

    # Tools whose results MUST NOT be cached (writes, mutations, side effects).
    write_tools: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> MCPServerConfig:
        if self.transport == "stdio":
            if not self.command:
                raise ValueError(f"server {self.id!r}: stdio transport requires `command`")
        else:
            if not self.url:
                raise ValueError(f"server {self.id!r}: {self.transport} transport requires `url`")
        if self.auth_type == "api_key" and not self.api_key:
            raise ValueError(f"server {self.id!r}: auth_type=api_key requires `api_key`")
        if self.auth_type == "oauth" and not (
            self.oauth_token_url and self.oauth_client_id and self.oauth_client_secret
        ):
            raise ValueError(
                f"server {self.id!r}: auth_type=oauth requires "
                "oauth_token_url + oauth_client_id + oauth_client_secret"
            )
        return self

    def static_headers(self) -> dict[str, str]:
        """Headers known without any network call (extra + api_key bearer).

        OAuth bearer tokens are NOT here — they are fetched dynamically by the
        client via :mod:`backend.v.mcp.oauth` because they expire and refresh.
        """
        headers = dict(self.extra_headers)
        if self.auth_type == "api_key" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def parse_servers(servers_json: str) -> list[MCPServerConfig]:
    """Parse the ``MCP_SERVERS_JSON`` env var into a list of configs.

    Empty string and ``"[]"`` both yield an empty list (MCP disabled).
    """
    if not servers_json or servers_json.strip() == "":
        return []
    try:
        raw = json.loads(servers_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"MCP_SERVERS_JSON is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError("MCP_SERVERS_JSON must be a JSON array")
    return [MCPServerConfig(**entry) for entry in raw]
