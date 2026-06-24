"""Unit tests for :mod:`backend.v.agents.orchestrator`."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage

from backend.v.agents.orchestrator import (
    PARALLEL_SAFE,
    ToolOrchestrator,
    _envelope,
    _wrap,
)


def _tc(name: str, tid: str = "c1", args: dict | None = None) -> dict:
    return {"id": tid, "name": name, "args": args or {}, "type": "tool_call"}


def _tool_msg(content: str, tid: str = "c1", status: str = "success") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tid, status=status)


def _fake_node(return_content: str, status: str = "success") -> AsyncMock:
    node = AsyncMock()
    node.ainvoke = AsyncMock(return_value={"messages": [_tool_msg(return_content, status=status)]})
    return node


class TestEnvelope:
    def test_ok_shape(self) -> None:
        out = _envelope("search", ok=True, data="结果", error=None)
        import json

        d = json.loads(out)
        assert d == {"ok": True, "tool": "search", "data": "结果", "error": None}

    def test_error_shape(self) -> None:
        out = _envelope("search", ok=False, data=None, error="失败原因")
        import json

        d = json.loads(out)
        assert d["ok"] is False and d["error"] == "失败原因"


class TestWrap:
    def test_wraps_success(self) -> None:
        msg = _tool_msg("hello")
        wrapped = _wrap(msg, "search")
        import json

        d = json.loads(wrapped.content)
        assert d["ok"] is True and d["data"] == "hello"
        assert wrapped.status == "success"

    def test_wraps_error(self) -> None:
        msg = _tool_msg("boom", status="error")
        wrapped = _wrap(msg, "calculator")
        import json

        d = json.loads(wrapped.content)
        assert d["ok"] is False and d["error"] == "boom"


class TestToolOrchestrator:
    def _make(self, permissions: dict | None = None) -> ToolOrchestrator:
        node = _fake_node("result")
        return ToolOrchestrator(node, None, permissions or {})

    def _state(self) -> dict:
        return {"messages": [AIMessage(content="", tool_calls=[])]}

    async def test_allowed_runs_and_wraps(self) -> None:
        node = _fake_node("检索结果ABC")
        orch = ToolOrchestrator(node, None, {"search": "allowed"})
        result = await orch.execute([_tc("search", "c1")], self._state(), {}, {})
        assert len(result.messages) == 1
        import json

        d = json.loads(result.messages[0].content)
        assert d["ok"] is True and "ABC" in d["data"]

    async def test_deny_blocks_without_executing(self) -> None:
        node = MagicMock()
        orch = ToolOrchestrator(node, None, {"calculator": "deny"})
        result = await orch.execute([_tc("calculator", "c1")], self._state(), {}, {})
        node.ainvoke.assert_not_called()
        import json

        d = json.loads(result.messages[0].content)
        assert d["ok"] is False
        assert result.messages[0].status == "error"

    async def test_ask_blocks_with_confirm_message(self) -> None:
        node = MagicMock()
        orch = ToolOrchestrator(node, None, {"subagent": "ask"})
        result = await orch.execute([_tc("subagent", "c1")], self._state(), {}, {})
        node.ainvoke.assert_not_called()
        import json

        d = json.loads(result.messages[0].content)
        assert d["ok"] is False
        assert "确认" in d["error"]

    async def test_absent_permission_defaults_to_allowed(self) -> None:
        node = _fake_node("ok")
        orch = ToolOrchestrator(node, None, {})  # no explicit permission
        result = await orch.execute([_tc("search", "c1")], self._state(), {}, {})
        assert result.messages[0].status == "success"

    async def test_parallel_safe_tool_goes_to_concurrent_node(self) -> None:
        concurrent = _fake_node("parallel_result")
        orch = ToolOrchestrator(concurrent, None, {})
        assert "search" in PARALLEL_SAFE
        result = await orch.execute([_tc("search", "c1")], self._state(), {}, {})
        concurrent.ainvoke.assert_called_once()
        assert result.messages

    async def test_batch_timeout_returns_error_envelope(self) -> None:
        import asyncio

        async def _hang(*a, **kw):
            await asyncio.sleep(9999)

        node = AsyncMock()
        node.ainvoke = _hang
        orch = ToolOrchestrator(node, None, {})
        with patch("backend.v.agents.orchestrator.TOOL_BATCH_TIMEOUT_SECONDS", 0.01):
            result = await orch.execute([_tc("search", "c1")], self._state(), {}, {})
        import json

        d = json.loads(result.messages[0].content)
        assert d["ok"] is False and "超时" in d["error"]

    async def test_error_count_updated_for_executed_calls(self) -> None:
        node = _fake_node("[tool_error] boom", status="error")
        orch = ToolOrchestrator(node, None, {})
        result = await orch.execute([_tc("search", "c1")], self._state(), {}, {})
        assert result.new_error_counts.get("search", 0) == 1

    async def test_denied_call_does_not_increment_error_count(self) -> None:
        node = MagicMock()
        orch = ToolOrchestrator(node, None, {"search": "deny"})
        result = await orch.execute([_tc("search", "c1")], self._state(), {}, {})
        assert result.new_error_counts.get("search", 0) == 0


class TestCollectToolFacts:
    """Test the updated _collect_tool_facts in intent_reflect."""

    def test_extracts_data_from_ok_envelope(self) -> None:
        import json

        from backend.v.agents.intent_reflect import _collect_tool_facts

        content = json.dumps({"ok": True, "tool": "search", "data": "商品结果", "error": None})
        msgs = [ToolMessage(content=content, tool_call_id="c1")]
        assert "商品结果" in _collect_tool_facts(msgs)

    def test_skips_error_envelope(self) -> None:
        import json

        from backend.v.agents.intent_reflect import _collect_tool_facts

        content = json.dumps({"ok": False, "tool": "search", "data": None, "error": "失败"})
        msgs = [ToolMessage(content=content, tool_call_id="c1")]
        out = _collect_tool_facts(msgs)
        assert "失败" not in out  # error text not surfaced as fact

    def test_backward_compat_plain_string(self) -> None:
        from backend.v.agents.intent_reflect import _collect_tool_facts

        msgs = [ToolMessage(content="普通结果字符串", tool_call_id="c1")]
        assert "普通结果字符串" in _collect_tool_facts(msgs)

    def test_guard_block_still_skipped(self) -> None:
        from backend.v.agents.intent_reflect import _collect_tool_facts

        msgs = [ToolMessage(content="[tool_guard] 超出预算", tool_call_id="c1")]
        out = _collect_tool_facts(msgs)
        assert out == "（本轮无工具调用结果，回复不应引用任何具体事实。）"
