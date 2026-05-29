"""Unit tests for ``backend.v.agents.nodes``."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.v.agents.nodes import _system_prompt, agent_node, enter_node, exit_node
from backend.v.models.llm_caller import LLMResult


class TestSystemPrompt:
    def test_includes_channel(self) -> None:
        prompt = _system_prompt(None, "wecom")
        assert "wecom" in prompt

    def test_appends_profile_bits(self) -> None:
        prompt = _system_prompt(
            {"customer_name": "李伟", "member_level": "黄金", "preferred_language": "zh"},
            "wecom",
        )
        assert "李伟" in prompt
        assert "黄金" in prompt
        assert "zh" in prompt

    def test_empty_profile_dict(self) -> None:
        prompt = _system_prompt({}, "feishu")
        assert "feishu" in prompt
        # No profile values present → the customer_profile section is omitted.
        assert "<customer_profile>" not in prompt

    def test_renders_xml_structure(self) -> None:
        prompt = _system_prompt(None, "wecom")
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

    def test_profile_uses_xml_subtags(self) -> None:
        prompt = _system_prompt(
            {"customer_name": "张三", "member_level": "白银"},
            "wecom",
        )
        assert "<customer_profile>" in prompt
        assert "<name>张三</name>" in prompt
        assert "<member_level>白银</member_level>" in prompt


class TestEnterNode:
    async def test_prepends_system_message_on_first_turn(self) -> None:
        update = await enter_node(
            {
                "messages": [HumanMessage(content="hi")],
                "channel": "wecom",
                "user_profile": {"member_level": "gold"},
            },
            config={"configurable": {}},
        )
        assert "messages" in update
        assert isinstance(update["messages"][0], SystemMessage)
        assert "wecom" in update["messages"][0].content

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
