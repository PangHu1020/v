"""Tests for ``backend.v.utils.logging``."""

from __future__ import annotations

import json
import logging

import pytest
import structlog

from backend.v.utils.logging import bind_request, configure, get_logger


@pytest.fixture(autouse=True)
def _reset_structlog() -> None:
    """Reset structlog config and contextvars between tests so tests are hermetic."""
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()


class TestConfigure:
    def test_text_mode_emits_human_readable(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        configure(level="INFO", json=False)
        log = get_logger("test")
        log.info("hello", foo="bar")
        captured = capsys.readouterr().out
        assert "hello" in captured
        assert "foo" in captured
        assert "bar" in captured

    def test_json_mode_emits_parseable_json(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        configure(level="INFO", json=True)
        log = get_logger("test")
        log.info("hello", foo="bar")
        line = capsys.readouterr().out.strip().splitlines()[-1]
        payload = json.loads(line)
        assert payload["event"] == "hello"
        assert payload["foo"] == "bar"
        assert payload["level"] == "info"
        assert "timestamp" in payload

    def test_level_filters_below_threshold(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        configure(level="WARNING", json=True)
        log = get_logger("test")
        log.info("should not appear")
        log.warning("should appear")
        out = capsys.readouterr().out
        assert "should not appear" not in out
        assert "should appear" in out

    def test_invalid_level_falls_back_to_info(self) -> None:
        configure(level="not-a-level", json=True)
        assert logging.getLogger().level == logging.INFO


class TestBindRequest:
    def test_binds_fields_inside_block(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        configure(level="INFO", json=True)
        log = get_logger("test")
        with bind_request(
            request_id="req-1",
            channel="wecom",
            channel_user_id="ext-42",
            session_id="sess-x",
        ):
            log.info("inside")
        line = capsys.readouterr().out.strip().splitlines()[-1]
        payload = json.loads(line)
        assert payload["request_id"] == "req-1"
        assert payload["channel"] == "wecom"
        assert payload["channel_user_id"] == "ext-42"
        assert payload["session_id"] == "sess-x"

    def test_unbinds_after_block(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        configure(level="INFO", json=True)
        log = get_logger("test")
        with bind_request(request_id="req-1"):
            log.info("inside")
        log.info("outside")
        lines = capsys.readouterr().out.strip().splitlines()
        outside = json.loads(lines[-1])
        assert "request_id" not in outside

    def test_optional_fields_are_omitted(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        configure(level="INFO", json=True)
        log = get_logger("test")
        with bind_request(request_id="req-only"):
            log.info("event")
        line = capsys.readouterr().out.strip().splitlines()[-1]
        payload = json.loads(line)
        assert payload["request_id"] == "req-only"
        assert "channel" not in payload
        assert "channel_user_id" not in payload

    def test_extra_kwargs_are_bound(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        configure(level="INFO", json=True)
        log = get_logger("test")
        with bind_request(request_id="req-1", custom_field="custom_value"):
            log.info("event")
        line = capsys.readouterr().out.strip().splitlines()[-1]
        payload = json.loads(line)
        assert payload["custom_field"] == "custom_value"
