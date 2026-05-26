"""Unit tests for ``backend.v.agents.state``."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages

from backend.v.agents.state import CustomerServiceState


class TestAddMessagesReducer:
    def test_initial_population(self) -> None:
        existing: list = []
        new = [SystemMessage(content="rules"), HumanMessage(content="hello")]
        merged = add_messages(existing, new)
        assert len(merged) == 2
        assert merged[0].content == "rules"
        assert merged[1].content == "hello"

    def test_appends_to_existing_history(self) -> None:
        existing = [HumanMessage(content="hi")]
        new = [AIMessage(content="hello back")]
        merged = add_messages(existing, new)
        assert len(merged) == 2
        assert merged[0].content == "hi"
        assert merged[1].content == "hello back"

    def test_does_not_clobber(self) -> None:
        existing = [HumanMessage(content="first")]
        before_id = id(existing)
        merged = add_messages(existing, [AIMessage(content="second")])
        assert id(merged) != before_id
        assert len(existing) == 1  # caller's list untouched


class TestStateShape:
    def test_typed_dict_fields(self) -> None:
        state: CustomerServiceState = {
            "messages": [HumanMessage(content="hi")],
            "session_id": "s-1",
            "channel": "wecom",
            "channel_user_id": "ext-1",
            "user_profile": {"member_level": "gold"},
            "interrupt_payload": None,
        }
        assert state["session_id"] == "s-1"
        assert state["user_profile"] == {"member_level": "gold"}
        assert state["interrupt_payload"] is None

    def test_partial_update_pattern(self) -> None:
        # Nodes return only the keys they want to update; merging is done by
        # LangGraph at runtime. The state TypedDict is total=False so partial
        # dicts are valid.
        update: CustomerServiceState = {"messages": [AIMessage(content="reply")]}
        assert "session_id" not in update
        assert "messages" in update
