"""Extension config loader for ``.agent/config.json`` (MCP servers + skills + tools).

A single project-local JSON file declares the agent's *extensions* — the MCP
servers it connects to, the skill sources it loads, and per-tool execution
permissions (allowed / ask / deny).

Conventions:
- **Per-entry ``enable``**: MCP servers / skill sources carry ``enable``.
- **``${VAR}`` interpolation**: secrets stay in ``.env``.
- **Tool permissions**: ``tools.permissions`` maps tool names to execution policy.

Shape::

    {
      "mcp": {"servers": [...]},
      "skill": {"sources": [...]},
      "tools": {
        "permissions": {
          "search":          "allowed",
          "calculator":      "allowed",
          "recall_memory":   "allowed",
          "load_skill":      "allowed",
          "subagent":        "ask",
          "transfer_to_human": "allowed"
        }
      }
    }

Missing file → empty config; absent tool entry → ``"allowed"`` (safe default).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from backend.v.mcp.config import MCPServerConfig
from backend.v.utils.logging import get_logger

_log = get_logger("configs.agent_config")

_CONFIG_FILE_ENV = "AGENT_CONFIG_FILE"
_DEFAULT_CONFIG_FILE = ".agent/config.json"

_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

ToolPermission = Literal["allowed", "ask", "deny"]
_VALID_PERMISSIONS: frozenset[str] = frozenset({"allowed", "ask", "deny"})


@dataclass(frozen=True)
class SkillSource:
    """One enabled skill directory to load markdown SOPs from."""

    path: str
    name: str = ""


@dataclass
class AgentConfig:
    """Resolved, enable-filtered extension config."""

    mcp_servers: list[MCPServerConfig] = field(default_factory=list)
    skill_sources: list[SkillSource] = field(default_factory=list)
    tool_permissions: dict[str, ToolPermission] = field(default_factory=dict)
    """Per-tool execution policy. Absent entry defaults to ``"allowed"``."""


def _config_path() -> str:
    return os.environ.get(_CONFIG_FILE_ENV, _DEFAULT_CONFIG_FILE)


def _interpolate(value: Any, env: dict[str, str]) -> Any:
    """Recursively resolve ``${VAR}`` in strings; recurse into lists/dicts.

    A reference to an unset variable raises — silent empties would let a
    mis-spelled secret name produce a server that fails auth at runtime with no
    hint why. Lists and dicts are walked; non-strings pass through untouched.
    """
    if isinstance(value, str):

        def _sub(m: re.Match[str]) -> str:
            name = m.group(1)
            if name not in env:
                raise ValueError(
                    f".agent config references ${{{name}}} but it is not set in the environment"
                )
            return env[name]

        return _VAR_RE.sub(_sub, value)
    if isinstance(value, list):
        return [_interpolate(v, env) for v in value]
    if isinstance(value, dict):
        return {k: _interpolate(v, env) for k, v in value.items()}
    return value


def _enabled(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only entries whose ``enable`` is truthy; strip the key from each.

    ``enable`` is loader metadata, not a field of the downstream model, so it is
    removed before the dict is handed to ``MCPServerConfig`` / ``SkillSource``.
    """
    out: list[dict[str, Any]] = []
    for e in entries:
        e = dict(e)
        if e.pop("enable", False):
            out.append(e)
    return out


def load_agent_config(
    path: str | Path | None = None,
    *,
    env: dict[str, str] | None = None,
) -> AgentConfig:
    """Load ``.agent/config.json`` → enable-filtered, interpolated config.

    Args:
        path: Override config path (default: ``$AGENT_CONFIG_FILE`` or
            ``.agent/config.json``).
        env: Environment mapping for ``${VAR}`` resolution (default: os.environ).

    Returns:
        :class:`AgentConfig` with only enabled, fully-resolved entries. A
        missing file yields an empty config (both subsystems disabled).

    Raises:
        ValueError: malformed JSON, wrong shape, or an unset ``${VAR}``.
    """
    env = dict(os.environ) if env is None else env
    cfg_path = Path(path) if path is not None else Path(_config_path())
    if not cfg_path.is_file():
        _log.info("configs.agent_config.absent", path=str(cfg_path))
        return AgentConfig()

    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{cfg_path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path} must be a JSON object with 'mcp' / 'skill' keys")

    mcp_raw = (raw.get("mcp") or {}).get("servers") or []
    skill_raw = (raw.get("skill") or {}).get("sources") or []
    if not isinstance(mcp_raw, list):
        raise ValueError(f"{cfg_path}: mcp.servers must be a JSON array")
    if not isinstance(skill_raw, list):
        raise ValueError(f"{cfg_path}: skill.sources must be a JSON array")

    # Filter by enable BEFORE interpolation — disabled entries may have
    # placeholder secrets we don't need to validate/resolve.
    mcp_enabled = _enabled(mcp_raw)
    skill_enabled = _enabled(skill_raw)

    # Now interpolate only the enabled subset.
    mcp_enabled = _interpolate(mcp_enabled, env)
    skill_enabled = _interpolate(skill_enabled, env)

    mcp_servers = [MCPServerConfig(**e) for e in mcp_enabled]
    skill_sources = [SkillSource(path=e["path"], name=e.get("name", "")) for e in skill_enabled]

    # Tool permissions: plain string values, no secrets, no interpolation needed.
    raw_perms: dict[str, Any] = (raw.get("tools") or {}).get("permissions") or {}
    if not isinstance(raw_perms, dict):
        raise ValueError(f"{cfg_path}: tools.permissions must be a JSON object")
    tool_permissions: dict[str, ToolPermission] = {}
    for tool_name, perm in raw_perms.items():
        if perm not in _VALID_PERMISSIONS:
            raise ValueError(
                f"{cfg_path}: tools.permissions.{tool_name}={perm!r} must be one of "
                f"{sorted(_VALID_PERMISSIONS)}"
            )
        tool_permissions[str(tool_name)] = perm  # type: ignore[assignment]

    _log.info(
        "configs.agent_config.loaded",
        path=str(cfg_path),
        mcp_servers=len(mcp_servers),
        skill_sources=len(skill_sources),
        mcp_total=len(mcp_raw),
        skill_total=len(skill_raw),
        tool_permissions=tool_permissions,
    )
    return AgentConfig(
        mcp_servers=mcp_servers,
        skill_sources=skill_sources,
        tool_permissions=tool_permissions,
    )
