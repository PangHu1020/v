"""Unit tests for .agent/config.json loader (MCP + skill extension config)."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from backend.v.configs.agent_config import AgentConfig, SkillSource, load_agent_config


class TestLoadAgentConfig:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        cfg = load_agent_config(tmp_path / "nonexistent.json")
        assert cfg.mcp_servers == []
        assert cfg.skill_sources == []

    def test_empty_json_object_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        p.write_text("{}")
        cfg = load_agent_config(p)
        assert cfg.mcp_servers == []
        assert cfg.skill_sources == []

    def test_enable_false_filters_out_entries(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        p.write_text(
            json.dumps(
                {
                    "mcp": {
                        "servers": [
                            {"enable": True, "id": "a", "transport": "http", "url": "http://a"},
                            {"enable": False, "id": "b", "transport": "http", "url": "http://b"},
                        ]
                    },
                    "skill": {
                        "sources": [
                            {"enable": True, "path": "/x"},
                            {"enable": False, "path": "/y"},
                        ]
                    },
                }
            )
        )
        cfg = load_agent_config(p)
        assert len(cfg.mcp_servers) == 1
        assert cfg.mcp_servers[0].id == "a"
        assert len(cfg.skill_sources) == 1
        assert cfg.skill_sources[0].path == "/x"

    def test_enable_missing_treated_as_false(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        p.write_text(
            json.dumps(
                {
                    "mcp": {"servers": [{"id": "x", "transport": "stdio", "command": "x"}]},
                    "skill": {"sources": [{"path": "/z"}]},
                }
            )
        )
        cfg = load_agent_config(p)
        assert cfg.mcp_servers == []
        assert cfg.skill_sources == []

    def test_var_interpolation_from_env(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        p.write_text(
            json.dumps(
                {
                    "mcp": {
                        "servers": [
                            {
                                "enable": True,
                                "id": "shop",
                                "transport": "http",
                                "url": "${BASE_URL}/mcp",
                                "auth_type": "oauth",
                                "oauth_client_secret": "${SECRET}",
                                "oauth_token_url": "http://tok",
                                "oauth_client_id": "cli",
                            }
                        ]
                    }
                }
            )
        )
        env = {"BASE_URL": "http://localhost:9000", "SECRET": "s3cr3t"}
        cfg = load_agent_config(p, env=env)
        assert cfg.mcp_servers[0].url == "http://localhost:9000/mcp"
        assert cfg.mcp_servers[0].oauth_client_secret == "s3cr3t"

    def test_var_interpolation_in_nested_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        p.write_text(
            json.dumps(
                {
                    "mcp": {
                        "servers": [
                            {
                                "enable": True,
                                "id": "x",
                                "transport": "stdio",
                                "command": "node",
                                "env": {"TOKEN": "${MY_TOKEN}"},
                            }
                        ]
                    }
                }
            )
        )
        cfg = load_agent_config(p, env={"MY_TOKEN": "abc123"})
        assert cfg.mcp_servers[0].env == {"TOKEN": "abc123"}

    def test_unset_var_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        p.write_text(
            json.dumps(
                {
                    "mcp": {
                        "servers": [
                            {
                                "enable": True,
                                "id": "x",
                                "transport": "http",
                                "url": "${UNSET}",
                            }
                        ]
                    }
                }
            )
        )
        with pytest.raises(ValueError, match=r"\$\{UNSET\}"):
            load_agent_config(p, env={})

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        p.write_text("not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_agent_config(p)

    def test_non_dict_root_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        p.write_text("[]")
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_agent_config(p)

    def test_servers_not_list_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        p.write_text('{"mcp":{"servers":"bad"}}')
        with pytest.raises(ValueError, match=r"mcp\.servers must be a JSON array"):
            load_agent_config(p)

    def test_sources_not_list_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        p.write_text('{"skill":{"sources":"bad"}}')
        with pytest.raises(ValueError, match=r"skill\.sources must be a JSON array"):
            load_agent_config(p)

    def test_skill_source_name_optional(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        p.write_text(
            json.dumps(
                {
                    "skill": {
                        "sources": [
                            {"enable": True, "path": "/a"},
                            {"enable": True, "path": "/b", "name": "Custom"},
                        ]
                    }
                }
            )
        )
        cfg = load_agent_config(p)
        assert cfg.skill_sources[0].name == ""
        assert cfg.skill_sources[1].name == "Custom"

    def test_real_world_full_config(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        p.write_text(
            dedent(
                """\
                {
                  "mcp": {
                    "servers": [
                      {
                        "enable": true,
                        "id": "shop",
                        "transport": "http",
                        "url": "http://localhost:9100/mcp",
                        "description": "Shop backend",
                        "auth_type": "oauth",
                        "oauth_token_url": "http://localhost:9100/token",
                        "oauth_client_id": "agent",
                        "oauth_client_secret": "${SHOP_OAUTH_SECRET}",
                        "oauth_scope": "tools:call",
                        "write_tools": ["create_handoff_ticket"]
                      },
                      {
                        "enable": false,
                        "id": "weather",
                        "transport": "stdio",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-weather"]
                      }
                    ]
                  },
                  "skill": {
                    "sources": [
                      {
                        "enable": true,
                        "path": "/srv/skills/internal",
                        "name": "Internal SOPs"
                      }
                    ]
                  }
                }
                """
            )
        )
        cfg = load_agent_config(p, env={"SHOP_OAUTH_SECRET": "secret123"})
        assert len(cfg.mcp_servers) == 1  # only enabled
        assert cfg.mcp_servers[0].id == "shop"
        assert cfg.mcp_servers[0].oauth_client_secret == "secret123"
        assert cfg.mcp_servers[0].write_tools == ["create_handoff_ticket"]
        assert len(cfg.skill_sources) == 1
        assert cfg.skill_sources[0].path == "/srv/skills/internal"
        assert cfg.skill_sources[0].name == "Internal SOPs"


class TestSkillSource:
    def test_dataclass_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        s = SkillSource(path="/x")
        with pytest.raises(FrozenInstanceError):
            s.path = "/y"  # type: ignore[misc]


class TestAgentConfig:
    def test_defaults_to_empty(self) -> None:
        cfg = AgentConfig()
        assert cfg.mcp_servers == []
        assert cfg.skill_sources == []
