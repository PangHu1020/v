"""Unit tests for ``backend.v.tools.subagent``."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.v.models.llm_caller import LLMResult
from backend.v.tools.subagent import (
    DEFAULT_SHARED_TURN_COUNT,
    _slice_parent_messages,
    subagent,
)


def _llm_returning(text: str) -> MagicMock:
    caller = MagicMock()
    caller.chat = AsyncMock(
        return_value=LLMResult(
            message=AIMessage(content=text),
            model="deepseek-flash",
            role="main_fallback",
            fallback_used=False,
            latency_ms=1,
        )
    )
    return caller


class TestSliceParentMessages:
    def test_skips_system_messages(self) -> None:
        state = {
            "messages": [
                SystemMessage(content="prompt"),
                HumanMessage(content="hi"),
                AIMessage(content="hello"),
            ]
        }
        out = _slice_parent_messages(state, limit=10)
        assert all(not isinstance(m, SystemMessage) for m in out)
        assert len(out) == 2

    def test_takes_tail_only(self) -> None:
        state = {"messages": [HumanMessage(content=f"m-{i}") for i in range(20)]}
        out = _slice_parent_messages(state, limit=5)
        assert len(out) == 5
        assert out[0].content == "m-15"
        assert out[-1].content == "m-19"

    def test_zero_limit(self) -> None:
        state = {"messages": [HumanMessage(content="x")]}
        assert _slice_parent_messages(state, limit=0) == []

    def test_missing_messages_key(self) -> None:
        assert _slice_parent_messages({}, limit=5) == []

    def test_non_dict_state(self) -> None:
        assert _slice_parent_messages("not a dict", limit=5) == []  # type: ignore[arg-type]


class TestSubagentTool:
    async def test_independent_mode_default(self) -> None:
        caller = _llm_returning("子任务结果")
        out = await subagent.ainvoke(
            {
                "task": "把这段话翻译成英文：你好，世界",
                "state": {"messages": []},
            },
            config={"configurable": {"llm_caller": caller}},
        )
        assert out == "子任务结果"

        msgs = caller.chat.call_args.args[1]
        # System prompt + the task HumanMessage; no parent context.
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[1], HumanMessage)
        assert "翻译" in msgs[1].content
        assert len(msgs) == 2

    async def test_shared_mode_includes_parent_messages(self) -> None:
        caller = _llm_returning("总结完成")
        parent_state = {
            "messages": [
                SystemMessage(content="客服 prompt"),
                HumanMessage(content="客户：订单 ORD123 在哪"),
                AIMessage(content="助手：我帮您查"),
                HumanMessage(content="客户：还要一份发票"),
            ]
        }
        await subagent.ainvoke(
            {
                "task": "总结到目前为止的客户诉求",
                "state": parent_state,
                "context_mode": "shared",
            },
            config={"configurable": {"llm_caller": caller}},
        )
        msgs = caller.chat.call_args.args[1]
        # System + parent body (3 non-system msgs) + task = 5 total.
        assert len(msgs) == 5
        # Parent's HumanMessage about order is forwarded.
        assert any("ORD123" in m.content for m in msgs if isinstance(m, HumanMessage))
        # Original parent system prompt is NOT included.
        assert sum(isinstance(m, SystemMessage) for m in msgs) == 1

    async def test_shared_mode_truncates_to_default_limit(self) -> None:
        caller = _llm_returning("ok")
        many = {"messages": [HumanMessage(content=f"m-{i}") for i in range(30)]}
        await subagent.ainvoke(
            {
                "task": "summarize",
                "state": many,
                "context_mode": "shared",
            },
            config={"configurable": {"llm_caller": caller}},
        )
        msgs = caller.chat.call_args.args[1]
        # 1 system + DEFAULT_SHARED_TURN_COUNT parent + 1 task
        assert len(msgs) == 1 + DEFAULT_SHARED_TURN_COUNT + 1

    async def test_missing_llm_caller_returns_error(self) -> None:
        out = await subagent.ainvoke(
            {"task": "x", "state": {}},
            config={"configurable": {}},
        )
        assert out.startswith("[subagent_error]")
        assert "llm_caller" in out

    async def test_llm_failure_returns_error(self) -> None:
        caller = MagicMock()
        caller.chat = AsyncMock(side_effect=RuntimeError("api down"))
        out = await subagent.ainvoke(
            {"task": "x", "state": {}},
            config={"configurable": {"llm_caller": caller}},
        )
        assert out.startswith("[subagent_error]")
        assert "RuntimeError" in out

    async def test_independent_mode_does_not_leak_parent_state(self) -> None:
        caller = _llm_returning("ok")
        parent = {
            "messages": [
                HumanMessage(content="客户问敏感内容"),
                AIMessage(content="助手：我处理一下"),
            ]
        }
        await subagent.ainvoke(
            {
                "task": "对「电池」做名词解释",
                "state": parent,
                "context_mode": "independent",
            },
            config={"configurable": {"llm_caller": caller}},
        )
        msgs = caller.chat.call_args.args[1]
        # Independent mode -> no parent messages forwarded.
        assert all(
            "客户问敏感内容" not in (m.content if isinstance(m.content, str) else "") for m in msgs
        )

    async def test_uses_main_fallback_role(self) -> None:
        caller = _llm_returning("ok")
        await subagent.ainvoke(
            {"task": "x", "state": {}},
            config={"configurable": {"llm_caller": caller}},
        )
        # Subagent uses the cheaper flash tier.
        role = caller.chat.call_args.args[0]
        assert role == "main_fallback"
