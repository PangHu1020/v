"""Unit tests for memory V2 types (``backend.v.memory.types``)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from backend.v.memory.types import (
    MemoryCandidate,
    MemoryExtraction,
    UserMemory,
)


class TestUserMemory:
    def test_minimal_construction_defaults(self) -> None:
        m = UserMemory(attr_key="preferred_courier", attr_value="顺丰", kind="preference")
        assert m.status == "active"
        assert m.source == "inferred"
        assert m.confidence == 0.5
        assert m.valid_to is None
        assert isinstance(m.valid_from, datetime)

    def test_value_accepts_scalar_and_list(self) -> None:
        scalar = UserMemory(attr_key="member_level", attr_value="黄金", kind="preference")
        listed = UserMemory(
            attr_key="risk_flags", attr_value=["易投诉", "高价值"], kind="constraint"
        )
        assert scalar.attr_value == "黄金"
        assert listed.attr_value == ["易投诉", "高价值"]

    def test_superseded_row_carries_valid_to(self) -> None:
        ts = datetime(2026, 6, 10, tzinfo=UTC)
        m = UserMemory(
            attr_key="preferred_courier",
            attr_value="京东",
            kind="preference",
            status="superseded",
            valid_to=ts,
        )
        assert m.status == "superseded"
        assert m.valid_to == ts

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValidationError):
            UserMemory(attr_key="x", attr_value="y", kind="event")  # type: ignore[arg-type]

    def test_rejects_bad_confidence(self) -> None:
        with pytest.raises(ValidationError):
            UserMemory(attr_key="x", attr_value="y", kind="preference", confidence=1.5)

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            UserMemory(
                attr_key="x",
                attr_value="y",
                kind="preference",
                bogus="z",  # type: ignore[call-arg]
            )


class TestMemoryCandidate:
    def test_semantic_candidate_shape(self) -> None:
        c = MemoryCandidate(
            content="客户偏好顺丰快递",
            importance=0.6,
            source="stated",
            confidence=0.9,
            attr_key="preferred_courier",
            attr_value="顺丰",
            kind="preference",
        )
        assert c.attr_key == "preferred_courier"
        assert c.subject is None

    def test_episodic_candidate_shape(self) -> None:
        c = MemoryCandidate(
            content="客户对订单 SO123 申请了退款",
            importance=0.7,
            subject="SO123",
        )
        # Episodic leaves the semantic fields empty.
        assert c.attr_key is None
        assert c.kind is None
        assert c.subject == "SO123"

    def test_defaults(self) -> None:
        c = MemoryCandidate(content="x", importance=0.3)
        assert c.source == "inferred"
        assert c.confidence == 0.5
        assert c.keywords == []

    def test_rejects_empty_content(self) -> None:
        with pytest.raises(ValidationError):
            MemoryCandidate(content="", importance=0.3)


class TestMemoryExtraction:
    def test_empty_default(self) -> None:
        e = MemoryExtraction()
        assert e.user_candidates == []
        assert e.episodic_candidates == []

    def test_routes_candidates_into_tiers(self) -> None:
        e = MemoryExtraction(
            user_candidates=[
                MemoryCandidate(
                    content="偏好顺丰",
                    importance=0.6,
                    attr_key="preferred_courier",
                    attr_value="顺丰",
                    kind="preference",
                )
            ],
            episodic_candidates=[
                MemoryCandidate(content="投诉了物流", importance=0.5, subject="物流")
            ],
        )
        assert len(e.user_candidates) == 1
        assert len(e.episodic_candidates) == 1
        assert e.user_candidates[0].attr_key == "preferred_courier"

    def test_roundtrip_json(self) -> None:
        e = MemoryExtraction(episodic_candidates=[MemoryCandidate(content="x", importance=0.2)])
        restored = MemoryExtraction.model_validate_json(e.model_dump_json())
        assert restored.episodic_candidates[0].content == "x"
