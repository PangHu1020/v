"""Unit tests for the ``load_skill`` progressive-disclosure tool."""

from __future__ import annotations

from backend.v.skills.model import Skill
from backend.v.skills.registry import SkillRegistry
from backend.v.tools.load_skill import load_skill


def _cfg(registry: SkillRegistry | None, channel: str | None = "wecom") -> dict:
    return {"configurable": {"skill_registry": registry, "channel": channel}}


class TestLoadSkill:
    async def test_returns_sop_body_for_known_skill(self) -> None:
        s = Skill(
            name="refund", description="退款流程", body="1. 询问订单号\n2. 核对金额", intents=[]
        )
        out = await load_skill.ainvoke({"name": "refund"}, config=_cfg(SkillRegistry([s])))
        assert '<sop name="refund">' in out
        assert "询问订单号" in out
        assert "核对金额" in out

    async def test_unknown_skill_returns_not_found(self) -> None:
        out = await load_skill.ainvoke({"name": "ghost"}, config=_cfg(SkillRegistry([])))
        assert "未找到" in out
        assert "ghost" in out

    async def test_missing_registry_returns_context_error(self) -> None:
        out = await load_skill.ainvoke({"name": "refund"}, config={"configurable": {}})
        assert "缺少运行上下文" in out

    async def test_channel_scoping_hides_other_channel_skill(self) -> None:
        s = Skill(name="feishu_only", body="飞书专用", intents=[], channels=["feishu"])
        out = await load_skill.ainvoke(
            {"name": "feishu_only"}, config=_cfg(SkillRegistry([s]), channel="wecom")
        )
        assert "未找到" in out
