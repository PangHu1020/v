"""Unit tests for ``backend.v.agents.compression_node``."""

from __future__ import annotations

from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

from backend.v.agents.compression_node import compression_node
from backend.v.memory.types import ConversationState, ExtractionResult, MemoryEntry
from backend.v.models.llm_caller import LLMResult


def _result(parsed: ExtractionResult) -> LLMResult:
    return LLMResult(
        message=AIMessage(content="{}"),
        model="test",
        role="memory_extract",
        fallback_used=False,
        latency_ms=1,
        parsed=parsed,
    )


def _long_history(n: int) -> list:
    msgs: list = []
    for i in range(n):
        msgs.append(HumanMessage(content=f"客户消息 {i} " * 50, id=f"h{i}"))
        msgs.append(AIMessage(content=f"助理回复 {i} " * 50, id=f"a{i}"))
    return msgs


class TestCompressionGuards:
    async def test_no_op_when_threshold_zero(self) -> None:
        out = await compression_node(
            {"messages": _long_history(10)},
            config={"configurable": {"compression_threshold_tokens": 0}},
        )
        assert out == {}

    async def test_no_op_when_below_token_threshold(self) -> None:
        out = await compression_node(
            {"messages": [HumanMessage(content="hi", id="h0")]},
            config={"configurable": {"compression_threshold_tokens": 100000}},
        )
        assert out == {}

    async def test_no_op_when_too_few_messages(self) -> None:
        out = await compression_node(
            {"messages": [HumanMessage(content="hi", id="h0")]},
            config={
                "configurable": {
                    "compression_threshold_tokens": 1,
                    "compression_keep_recent_messages": 4,
                }
            },
        )
        assert out == {}


class TestCompressionApplies:
    async def test_trims_head_and_injects_structured_summary(self) -> None:
        history = _long_history(8)  # 16 messages
        state_summary = ConversationState(
            current_topic="退款进度",
            events=["客户咨询订单 SO123"],
            actions_taken=["已查询订单"],
            key_facts=["订单号 SO123"],
        )
        caller = AsyncMock()
        caller.chat = AsyncMock(
            return_value=_result(
                ExtractionResult(
                    conversation_state=state_summary,
                    working_memories=[],
                    event_memories=[],
                )
            )
        )
        out = await compression_node(
            {
                "messages": history,
                "session_id": "sess-1",
                "channel": "wecom",
                "channel_user_id": "u1",
            },
            config={
                "configurable": {
                    "compression_threshold_tokens": 1,
                    "compression_keep_recent_messages": 4,
                    "llm_caller": caller,
                }
            },
        )
        update = out["messages"]
        removals = [m for m in update if isinstance(m, RemoveMessage)]
        systems = [m for m in update if isinstance(m, SystemMessage)]
        # 16 - 4 kept = 12 head messages removed.
        assert len(removals) == 12
        assert len(systems) == 1
        assert "<compressed_history>" in systems[0].content
        assert "退款进度" in systems[0].content
        assert "SO123" in systems[0].content
        # Last 4 messages preserved verbatim.
        kept_ids = {m.id for m in update if isinstance(m, (HumanMessage, AIMessage))}
        assert {"h6", "a6", "h7", "a7"} == kept_ids

    async def test_extraction_failure_still_trims(self) -> None:
        history = _long_history(8)
        caller = AsyncMock()
        caller.chat = AsyncMock(side_effect=RuntimeError("llm down"))
        out = await compression_node(
            {"messages": history, "session_id": "s", "channel": "wecom", "channel_user_id": "u"},
            config={
                "configurable": {
                    "compression_threshold_tokens": 1,
                    "compression_keep_recent_messages": 4,
                    "llm_caller": caller,
                }
            },
        )
        # Even with no parsed state, the head is still trimmed + a marker injected.
        systems = [m for m in out["messages"] if isinstance(m, SystemMessage)]
        assert len(systems) == 1
        assert "<compressed_history>" in systems[0].content

    async def test_folds_memories_into_working_no_pg_write(self, monkeypatch) -> None:
        history = _long_history(8)
        working = [MemoryEntry(content="本次偏好顺丰", importance=0.6, kind="preference")]
        events = [MemoryEntry(content="投诉 SO123 延误", importance=0.5, kind="event")]
        caller = AsyncMock()
        caller.chat = AsyncMock(
            return_value=_result(
                ExtractionResult(
                    conversation_state=ConversationState(current_topic="物流"),
                    working_memories=working,
                    event_memories=events,
                )
            )
        )

        append_mock = AsyncMock()
        insert_mock = AsyncMock()
        read_mock = AsyncMock(return_value=[*working, *events])
        monkeypatch.setattr("backend.v.memory.working.append_working_memory", append_mock)
        monkeypatch.setattr("backend.v.memory.working.read_working_memory", read_mock)
        monkeypatch.setattr("backend.v.memory.event_memory.insert_event_memories", insert_mock)

        out = await compression_node(
            {
                "messages": history,
                "session_id": "sess-w",
                "channel": "wecom",
                "channel_user_id": "u1",
            },
            config={
                "configurable": {
                    "compression_threshold_tokens": 1,
                    "compression_keep_recent_messages": 4,
                    "llm_caller": caller,
                    "redis": object(),
                }
            },
        )
        # V2: both working- and event-kind sentences fold into Redis in one
        # append; NO mid-session PG write (durable writes are session-end only).
        append_mock.assert_awaited_once()
        folded = append_mock.await_args.kwargs["entries"]
        assert len(folded) == 2  # working + event sentences combined
        insert_mock.assert_not_awaited()
        # The re-read working memory is rendered into the compressed marker.
        systems = [m for m in out["messages"] if isinstance(m, SystemMessage)]
        assert "本次偏好顺丰" in systems[0].content
