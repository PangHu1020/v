"""Unit tests for ``backend.v.skills.registry``."""

from __future__ import annotations

from backend.v.skills.model import Skill
from backend.v.skills.registry import SkillRegistry


def _skill(
    name: str,
    *,
    intents: list[str] | None = None,
    priority: int = 3,
    channels: list[str] | None = None,
    body: str = "body content",
) -> Skill:
    return Skill(
        name=name,
        intents=intents or [],
        priority=priority,
        channels=channels,
        body=body,
    )


class TestGet:
    def test_returns_skill_by_exact_name(self) -> None:
        s = _skill("refund", intents=["退款"], body="退款步骤")
        assert SkillRegistry([s]).get("refund") is s

    def test_unknown_name_returns_none(self) -> None:
        assert SkillRegistry([_skill("refund")]).get("nope") is None

    def test_channel_filter_excludes(self) -> None:
        reg = SkillRegistry([_skill("only_wecom", channels=["wecom"])])
        assert reg.get("only_wecom", channel="feishu") is None
        assert reg.get("only_wecom", channel="wecom") is not None

    def test_channel_none_ignores_filter(self) -> None:
        reg = SkillRegistry([_skill("only_wecom", channels=["wecom"])])
        assert reg.get("only_wecom") is not None


class TestCatalog:
    def test_lists_all_channel_skills_regardless_of_query(self) -> None:
        reg = SkillRegistry([_skill("a", intents=[]), _skill("b", intents=["退款"])])
        # match() would drop "a" (no intents); catalog lists both.
        assert {s.name for s in reg.catalog(channel="wecom")} == {"a", "b"}

    def test_channel_restriction_filters(self) -> None:
        reg = SkillRegistry(
            [
                _skill("w", channels=["wecom"]),
                _skill("f", channels=["feishu"]),
            ]
        )
        assert [s.name for s in reg.catalog(channel="wecom")] == ["w"]

    def test_render_catalog_emits_schema_not_body(self) -> None:
        s = Skill(name="refund", description="退款流程", body="机密正文步骤", intents=[])
        out = SkillRegistry([s]).render_catalog(channel="wecom")
        assert "<available_skills>" not in out  # caller wraps; renderer emits inner
        assert 'name="refund"' in out
        assert "退款流程" in out
        assert "机密正文步骤" not in out  # body is NOT disclosed in the catalog

    def test_render_catalog_empty_when_no_skills(self) -> None:
        assert SkillRegistry([]).render_catalog(channel="wecom") == ""


class TestRegistryProperties:
    def test_skills_accessor_is_a_copy(self) -> None:
        s = _skill("x", intents=["x"])
        reg = SkillRegistry([s])
        out = reg.skills
        out.clear()
        assert len(reg) == 1  # internal list untouched

    def test_len(self) -> None:
        assert len(SkillRegistry([])) == 0
        assert len(SkillRegistry([_skill("x"), _skill("y")])) == 2
