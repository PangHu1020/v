"""MCP server registry + LangChain tool wrapper.

The registry owns the per-server :class:`MCPClient` lifecycle and exposes
their tools to the agent as a flat list of :class:`StructuredTool`.

Tool naming: ``{server_id}__{tool_name}`` so the same tool name can come
from multiple servers without colliding (e.g., ``products__search`` and
``ads__search``). The agent sees each tool by its prefixed name.

Caching: results of non-write tools go through :class:`MCPToolCache`.
A tool is "write-class" if its name appears in the server's
``write_tools`` config list.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, ConfigDict, Field, create_model

from backend.v.mcp.cache import MCPToolCache
from backend.v.mcp.client import MCPClient, MCPClientError
from backend.v.mcp.config import MCPServerConfig
from backend.v.utils.logging import bind_request, get_logger

_log = get_logger("mcp.registry")

_JSON_TYPE_TO_PY: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _python_type_for(prop: dict[str, Any]) -> type:
    """Map a JSON-schema ``type`` to a Python annotation. Falls back to ``str``."""
    json_type = prop.get("type", "string")
    if isinstance(json_type, list):
        # Take the first non-null type for a union of simple types.
        json_type = next((t for t in json_type if t != "null"), "string")
    return _JSON_TYPE_TO_PY.get(json_type, str)


def _build_args_model(tool_name: str, input_schema: dict[str, Any] | None) -> type[BaseModel]:
    """Build a Pydantic model that mirrors the tool's JSON-schema input.

    Best-effort; nested objects and constrained types fall back to plain
    ``str`` / ``dict``. The LLM still sees a correct top-level JSON schema
    via LangChain's tool binding because ``StructuredTool`` re-emits it.
    """
    if not input_schema:
        return create_model(  # type: ignore[call-overload]
            f"{tool_name}_Args",
            __config__=ConfigDict(extra="allow"),
        )
    props = input_schema.get("properties") or {}
    required = set(input_schema.get("required") or [])
    fields: dict[str, tuple[Any, Any]] = {}
    for key, prop in props.items():
        py_type = _python_type_for(prop)
        description = prop.get("description", "")
        if key in required:
            fields[key] = (py_type, Field(..., description=description))
        else:
            fields[key] = (py_type | None, Field(default=None, description=description))
    return create_model(  # type: ignore[call-overload]
        f"{tool_name}_Args",
        __config__=ConfigDict(extra="allow"),
        **fields,
    )


def _extract_text(result: CallToolResult) -> str:
    """Flatten an MCP tool result into a single string the agent can read."""
    if result.isError:
        # Surface the error text but don't crash; the agent can decide.
        parts = [c.text for c in result.content if isinstance(c, TextContent)]
        return f"[tool_error] {' '.join(parts) or 'unknown error'}"
    parts: list[str] = []
    for c in result.content:
        if isinstance(c, TextContent):
            parts.append(c.text)
        else:
            dumper = getattr(c, "model_dump_json", None)
            parts.append(dumper() if callable(dumper) else str(c))
    return "\n".join(parts).strip()


class MCPRegistry:
    """Owns the lifecycle of all configured MCP clients + builds LangChain tools."""

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
        self._clients: dict[str, MCPClient] = {}
        self._tools: list[StructuredTool] = []

    @property
    def clients(self) -> dict[str, MCPClient]:
        return dict(self._clients)

    @property
    def tools(self) -> list[StructuredTool]:
        return list(self._tools)

    async def connect_all(self) -> None:
        """Connect every configured server and discover its tools."""
        for cfg in self._configs:
            client = MCPClient(cfg, call_timeout_seconds=self._call_timeout)
            try:
                await client.connect()
            except Exception as exc:
                _log.error("mcp.registry.connect_failed", server_id=cfg.id, error=str(exc))
                continue
            self._clients[cfg.id] = client

            try:
                listing = await client.list_tools()
            except Exception as exc:
                _log.error(
                    "mcp.registry.list_tools_failed",
                    server_id=cfg.id,
                    error=str(exc),
                )
                continue
            for mcp_tool in listing.tools:
                lc_tool = self._build_langchain_tool(client, mcp_tool)
                self._tools.append(lc_tool)
                _log.info(
                    "mcp.registry.tool_registered",
                    server_id=cfg.id,
                    tool_name=mcp_tool.name,
                    qualified=lc_tool.name,
                )

    async def aclose(self) -> None:
        """Close all clients. Safe to call multiple times."""
        for cid, client in list(self._clients.items()):
            try:
                await client.aclose()
            except Exception as exc:
                _log.warning("mcp.registry.close_error", server_id=cid, error=str(exc))
        self._clients.clear()

    def _build_langchain_tool(self, client: MCPClient, mcp_tool: Any) -> StructuredTool:
        cfg = client.config
        qualified_name = f"{cfg.id}__{mcp_tool.name}"
        is_write = mcp_tool.name in set(cfg.write_tools)
        args_schema = _build_args_model(qualified_name, mcp_tool.inputSchema)

        async def _run(**kwargs: Any) -> str:
            with bind_request(mcp_server=cfg.id, mcp_tool=mcp_tool.name):
                if not is_write:
                    cached = await self._cache.get(
                        server_id=cfg.id,
                        tool_name=mcp_tool.name,
                        arguments=kwargs,
                    )
                    if cached is not None:
                        return str(cached)
                try:
                    result = await client.call_tool(mcp_tool.name, kwargs)
                except MCPClientError as exc:
                    return f"[tool_error] {exc}"
                text = _extract_text(result)
                if not is_write and not text.startswith("[tool_error]"):
                    await self._cache.put(
                        server_id=cfg.id,
                        tool_name=mcp_tool.name,
                        arguments=kwargs,
                        value=text,
                    )
                return text

        description = mcp_tool.description or f"MCP tool {mcp_tool.name} on {cfg.id}"
        return StructuredTool.from_function(
            coroutine=_run,
            name=qualified_name,
            description=description,
            args_schema=args_schema,
        )


def serialize_tool_for_audit(tool: StructuredTool) -> str:
    """Helper used by tests / debugging to confirm a tool's surface."""
    schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
    return json.dumps(
        {"name": tool.name, "description": tool.description, "schema": schema},
        ensure_ascii=False,
    )
