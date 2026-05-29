"""Unit tests for ``backend.v.agents.nodes``."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.v.agents.nodes import _render_profile, agent_node, enter_node, exit_node
from backend.v.agents.prompts import build_main_system_prompt
from backend.v.memory.types import MemoryEntry
from backend.v.models.llm_caller import LLMResult


class TestSystemPromptStatic:
    """The static system prompt no longer interpolates profile.

    Profile rendering moved to ``_render_profile`` in ``nodes.py``;
    ``build_main_system_prompt`` only depends on the channel.
    """

    def test_includes_channel(self) -> None:
        prompt = build_main_system_prompt("wecom")
        assert "wecom" in prompt

    def test_renders_xml_structure(self) -> None:
        prompt = build_main_system_prompt("wecom")
        for tag in (
            "<role>",
            "</role>",
            "<task>",
            "<capabilities>",
            "<workflow>",
            "<style>",
            "<constraints>",
        ):
            assert tag in prompt

    def test_constraints_explain_layer_shadowing(self) -> None:
        prompt = build_main_system_prompt("wecom")
        # The shadowing rule needs to be visible to the model so we
        # don't accidentally drop it during a future prompt edit.
        assert "session_memory 覆盖 recent_events" in prompt
        assert "recent_events 覆盖 customer_profile" in prompt


class TestRenderProfile:
    def test_empty_profile_returns_empty_string(self) -> None:
        assert _render_profile(None) == ""
        assert _render_profile({}) == ""

    def test_canonical_fields_render_as_subtags(self) -> None:
        out = _render_profile(
            {
                "customer_name": "张三",
                "preferred_salutation": "先生",
                "member_level": "黄金",
            }
        )
        assert "<customer_profile>" in out
        assert "<customer_name>张三</customer_name>" in out
        assert "<preferred_salutation>先生</preferred_salutation>" in out
        assert "<member_level>黄金</member_level>" in out

    def test_extras_render_as_extra_tags(self) -> None:
        out = _render_profile(
            {
                "customer_name": "李四",
                "extras": {"preferred_courier": "顺丰", "preferred_size": "L"},
            }
        )
        assert "<extras>" in out
        assert '<extra key="preferred_courier">顺丰</extra>' in out
        assert '<extra key="preferred_size">L</extra>' in out

    def test_risk_flags_join(self) -> None:
        out = _render_profile({"risk_flags": ["易投诉", "高价值"]})
        assert "<risk_flags>易投诉,高价值</risk_flags>" in out

    def test_notes_render(self) -> None:
        out = _render_profile({"notes": "客户对物流敏感"})
        assert "<notes>客户对物流敏感</notes>" in out

    def test_unknown_keys_tolerated(self) -> None:
        # A legacy / pre-validated key falls back to the flat-dump renderer.
        out = _render_profile({"legacy_field": "value"})
        assert "<legacy_field>value</legacy_field>" in out


class TestEnterNode:
    async def test_prepends_system_message_on_first_turn(self) -> None:
        update = await enter_node(
            {
                "messages": [HumanMessage(content="hi")],
                "channel": "wecom",
                "user_profile": {"member_level": "黄金"},
            },
            config={"configurable": {}},
        )
        assert "messages" in update
        assert isinstance(update["messages"][0], SystemMessage)
        content = update["messages"][0].content
        assert "wecom" in content
        assert "<customer_profile>" in content
        assert "<member_level>黄金</member_level>" in content

    async def test_layers_recent_events_and_working_memory(self) -> None:
        recent = [
            {
                "content": "投诉过物流延误",
                "kind": "event",
                "importance": 0.7,
                "keywords": ["物流"],
                "created_at": datetime(2026, 5, 20, tzinfo=UTC),
            }
        ]
        working = [
            MemoryEntry(
                content="本次会话偏好顺丰",
                importance=0.6,
                keywords=["快递", "顺丰"],
                kind="preference",
            )
        ]
        update = await enter_node(
            {
                "messages": [HumanMessage(content="物流到哪了")],
                "channel": "wecom",
                "user_profile": {},
                "recent_events": recent,
                "working_memory": working,
            },
            config={"configurable": {}},
        )
        content = update["messages"][0].content
        assert "<recent_events>" in content
        assert "投诉过物流延误" in content
        assert "<session_memory>" in content
        assert "本次会话偏好顺丰" in content
        # Order: profile (empty here) → recent_events → session_memory.
        assert content.index("<recent_events>") < content.index("<session_memory>")

    async def test_does_not_double_prepend(self) -> None:
        update = await enter_node(
            {
                "messages": [SystemMessage(content="prior"), HumanMessage(content="hi")],
                "channel": "wecom",
            },
            config={"configurable": {}},
        )
        assert update == {}


class TestAgentNode:
    async def test_appends_ai_message_from_llm_caller(self) -> None:
        caller = AsyncMock()
        caller.chat = AsyncMock(
            return_value=LLMResult(
                message=AIMessage(content="hi back"),
                model="deepseek-flash",
                role="main_primary",
                fallback_used=False,
                latency_ms=42,
            )
        )

        update = await agent_node(
            {"messages": [HumanMessage(content="hello")]},
            {"configurable": {"llm_caller": caller}},
        )
        assert isinstance(update["messages"][0], AIMessage)
        assert update["messages"][0].content == "hi back"
        caller.chat.assert_awaited_once()

    async def test_missing_llm_caller_raises(self) -> None:
        with pytest.raises(RuntimeError, match="llm_caller"):
            await agent_node(
                {"messages": [HumanMessage(content="x")]},
                {"configurable": {}},
            )


class TestExitNode:
    async def test_returns_no_state_update(self) -> None:
        update = await exit_node({"session_id": "s-1", "messages": []})
        assert update == {}
