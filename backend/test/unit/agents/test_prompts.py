"""Unit tests for compression-summary rendering in ``backend.v.agents.prompts``."""

from __future__ import annotations

from backend.v.agents.prompts import build_compression_summary
from backend.v.memory.types import ConversationState


class TestBuildCompressionSummary:
    def test_empty_state_still_wraps_marker(self) -> None:
        out = build_compression_summary(conversation_state=None, session_memory_block="")
        assert "<compressed_history>" in out
        assert "</compressed_history>" in out
        # No state and no session memory → just the boundary + the note.
        assert "<conversation_state>" not in out

    def test_renders_all_state_fields(self) -> None:
        state = ConversationState(
            current_topic="退款进度",
            events=["客户咨询订单 SO123 退款"],
            actions_taken=["已查询订单状态"],
            unresolved_questions=["客户未确认退款方式"],
            key_facts=["订单号 SO123", "金额 ¥299"],
        )
        out = build_compression_summary(conversation_state=state, session_memory_block="")
        assert "<conversation_state>" in out
        assert "退款进度" in out
        assert "客户咨询订单 SO123 退款" in out
        assert "已查询订单状态" in out
        assert "客户未确认退款方式" in out
        assert "SO123" in out
        assert "¥299" in out

    def test_empty_state_omits_conversation_block_keeps_session_memory(self) -> None:
        out = build_compression_summary(
            conversation_state=ConversationState(),
            session_memory_block="<session_memory>...</session_memory>",
        )
        assert "<conversation_state>" not in out
        assert "<session_memory>" in out

    def test_partial_state_only_emits_present_sections(self) -> None:
        state = ConversationState(current_topic="物流查询")
        out = build_compression_summary(conversation_state=state, session_memory_block="")
        assert "物流查询" in out
        assert "<events>" not in out
        assert "<key_facts>" not in out
