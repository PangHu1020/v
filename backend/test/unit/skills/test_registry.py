"""Unit tests for ``backend.v.skills.registry``."""

from __future__ import annotations

import pytest

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


class TestMatch:
    def test_no_match_when_no_intents_overlap(self) -> None:
        reg = SkillRegistry([_skill("refund", intents=["退款"])])
        assert reg.match("我想问物流", channel="wecom") == []

    def test_match_by_intent_substring(self) -> None:
        reg = SkillRegistry([_skill("refund", intents=["退款"])])
        out = reg.match("怎么办理退款？", channel="wecom")
        assert len(out) == 1
        assert out[0].name == "refund"

    def test_case_insensitive(self) -> None:
        reg = SkillRegistry([_skill("refund", intents=["REFUND"])])
        out = reg.match("how do I refund this", channel="wecom")
        assert len(out) == 1

    def test_higher_match_count_ranks_first(self) -> None:
        reg = SkillRegistry(
            [
                _skill("a", intents=["退款"]),
                _skill("b", intents=["退款", "退货"]),
            ]
        )
        out = reg.match("我要退款，顺便退货", channel="wecom", top_k=2)
        assert [s.name for s in out] == ["b", "a"]

    def test_priority_breaks_ties(self) -> None:
        reg = SkillRegistry(
            [
                _skill("low", intents=["物流"], priority=1),
                _skill("high", intents=["物流"], priority=8),
            ]
        )
        out = reg.match("物流", channel="wecom", top_k=2)
        assert [s.name for s in out] == ["high", "low"]

    def test_top_k_zero_returns_empty(self) -> None:
        reg = SkillRegistry([_skill("x", intents=["x"])])
        assert reg.match("x", channel="wecom", top_k=0) == []

    def test_empty_query_returns_empty(self) -> None:
        reg = SkillRegistry([_skill("x", intents=["x"])])
        assert reg.match("", channel="wecom") == []

    def test_channel_restriction_excludes(self) -> None:
        reg = SkillRegistry([_skill("only_wecom", intents=["退款"], channels=["wecom"])])
        assert reg.match("退款", channel="feishu") == []
        assert len(reg.match("退款", channel="wecom")) == 1

    def test_channel_none_means_all_channels(self) -> None:
        reg = SkillRegistry([_skill("everywhere", intents=["退款"], channels=None)])
        assert len(reg.match("退款", channel="wecom")) == 1
        assert len(reg.match("退款", channel="feishu")) == 1

    def test_top_k_caps_results(self) -> None:
        reg = SkillRegistry([_skill(f"s{i}", intents=["退款"]) for i in range(5)])
        out = reg.match("退款", channel="wecom", top_k=2)
        assert len(out) == 2

    def test_skills_without_intents_never_match(self) -> None:
        reg = SkillRegistry([_skill("orphan", intents=[])])
        assert reg.match("anything", channel="wecom") == []


class TestRenderForPrompt:
    def test_empty(self) -> None:
        reg = SkillRegistry([])
        assert reg.render_for_prompt([]) == ""

    def test_single_skill(self) -> None:
        s = _skill("refund", body="1. 询问订单号")
        out = SkillRegistry([s]).render_for_prompt([s])
        assert "refund" in out
        assert "询问订单号" in out
        assert "SOP" in out  # The header line mentions SOP.

    def test_includes_description(self) -> None:
        s = Skill(name="x", description="退款流程", body="step 1", intents=[])
        out = SkillRegistry([s]).render_for_prompt([s])
        assert "退款流程" in out

    def test_multiple_in_priority_order(self) -> None:
        a = _skill("a", body="body-A")
        b = _skill("b", body="body-B")
        out = SkillRegistry([a, b]).render_for_prompt([a, b])
        # Both bodies appear, in the supplied order.
        assert out.index("body-A") < out.index("body-B")


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


@pytest.mark.parametrize(
    "query,channel,expected_names",
    [
        ("我想退款", "wecom", ["refund"]),
        ("查询物流情况", "wecom", ["logistics"]),
        ("退货并查物流", "wecom", ["logistics", "refund"]),  # logistics scores 2 (退 + 物流)
        ("无关问题", "wecom", []),
    ],
)
def test_realistic_routing(
    query: str,
    channel: str,
    expected_names: list[str],
) -> None:
    skills = [
        _skill("refund", intents=["退款", "退货"], priority=5),
        _skill("logistics", intents=["物流", "退货", "签收"], priority=3),
    ]
    reg = SkillRegistry(skills)
    out = reg.match(query, channel=channel, top_k=5)
    assert sorted(s.name for s in out) == sorted(expected_names)
