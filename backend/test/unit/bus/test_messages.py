"""Unit tests for ``backend.app.bus.messages``."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from backend.app.bus.messages import SystemMessage


def _kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "channel": "wecom",
        "channel_user_id": "ext-1",
        "text": "hello",
        "dedup_key": "msg-1",
    }
    base.update(overrides)
    return base


class TestSystemMessage:
    def test_minimal_fields(self) -> None:
        msg = SystemMessage(**_kwargs())
        assert msg.channel == "wecom"
        assert msg.attachments == []
        assert msg.raw_payload_ref is None
        assert msg.received_at is not None

    def test_unknown_channel_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SystemMessage(**_kwargs(channel="discord"))

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SystemMessage(**_kwargs(unexpected="bad"))

    def test_empty_dedup_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SystemMessage(**_kwargs(dedup_key=""))

    def test_json_round_trip(self) -> None:
        original = SystemMessage(
            **_kwargs(channel="feishu", attachments=[{"type": "image", "url": "http://x"}])
        )
        wire = original.model_dump_json()
        # Wire form is plain JSON.
        parsed = json.loads(wire)
        assert parsed["channel"] == "feishu"
        assert parsed["attachments"][0]["type"] == "image"
        # And the model can rehydrate from the same payload.
        rehydrated = SystemMessage.model_validate_json(wire)
        assert rehydrated == original

    def test_frozen(self) -> None:
        msg = SystemMessage(**_kwargs())
        with pytest.raises(ValidationError):
            msg.text = "mutated"  # type: ignore[misc]
