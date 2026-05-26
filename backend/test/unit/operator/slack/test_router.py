"""Unit tests for ``backend.app.operator.slack.router``."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from urllib.parse import urlencode

import fakeredis.aioredis
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.app.gateway.middleware import RequestIdMiddleware
from backend.app.operator.slack.router import build_router
from backend.app.operator.slack.signature import compute_signature

SECRET = "slack-test-secret"


class _RecordingHooks:
    def __init__(self) -> None:
        self.resumed: list[str] = []
        self.operator_messages: list[tuple[str, str]] = []

    async def on_resume(self, session_id: str) -> None:
        self.resumed.append(session_id)

    async def on_operator_message(self, session_id: str, text: str) -> None:
        self.operator_messages.append((session_id, text))


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


@pytest.fixture
async def stack(
    redis_client: fakeredis.aioredis.FakeRedis,
) -> AsyncIterator[tuple[AsyncClient, _RecordingHooks]]:
    hooks = _RecordingHooks()
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    app.include_router(
        build_router(
            signing_secret=SECRET,
            redis=redis_client,
            on_resume=hooks.on_resume,
            on_operator_message=hooks.on_operator_message,
        )
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c, hooks


def _signed(body: bytes, *, ts: str | None = None) -> tuple[bytes, dict[str, str]]:
    if ts is None:
        ts = str(int(time.time()))
    sig = compute_signature(SECRET, ts, body)
    return body, {
        "x-slack-request-timestamp": ts,
        "x-slack-signature": sig,
        "content-type": "application/json",
    }


class TestEvents:
    async def test_url_verification(self, stack: tuple[AsyncClient, _RecordingHooks]) -> None:
        c, _ = stack
        body, headers = _signed(
            json.dumps({"type": "url_verification", "challenge": "ch1"}).encode(),
        )
        resp = await c.post("/operator/slack/events", content=body, headers=headers)
        assert resp.status_code == 200
        assert resp.json() == {"challenge": "ch1"}

    async def test_thread_reply_invokes_callback(
        self,
        stack: tuple[AsyncClient, _RecordingHooks],
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        c, hooks = stack
        # Pre-warm thread mapping.
        await redis_client.set("slack_thread:1700.5", "sess-X", ex=3600)

        event = {
            "type": "event_callback",
            "event": {
                "type": "message",
                "thread_ts": "1700.5",
                "text": "请帮客户处理一下",
                "user": "U1",
                "ts": "1700.6",
            },
        }
        body, headers = _signed(json.dumps(event).encode())
        resp = await c.post("/operator/slack/events", content=body, headers=headers)
        assert resp.status_code == 200
        assert hooks.operator_messages == [("sess-X", "请帮客户处理一下")]

    async def test_bot_message_ignored(
        self,
        stack: tuple[AsyncClient, _RecordingHooks],
        redis_client: fakeredis.aioredis.FakeRedis,
    ) -> None:
        c, hooks = stack
        await redis_client.set("slack_thread:1700.5", "sess-X", ex=3600)
        event = {
            "type": "event_callback",
            "event": {
                "type": "message",
                "thread_ts": "1700.5",
                "text": "this is from a bot",
                "bot_id": "B123",  # bot-authored
                "ts": "1700.7",
            },
        }
        body, headers = _signed(json.dumps(event).encode())
        resp = await c.post("/operator/slack/events", content=body, headers=headers)
        assert resp.status_code == 200
        assert hooks.operator_messages == []

    async def test_top_level_message_ignored(
        self,
        stack: tuple[AsyncClient, _RecordingHooks],
    ) -> None:
        c, hooks = stack
        # No thread_ts → top-level channel message; not a thread reply.
        event = {
            "type": "event_callback",
            "event": {"type": "message", "text": "noise", "ts": "1700.8"},
        }
        body, headers = _signed(json.dumps(event).encode())
        resp = await c.post("/operator/slack/events", content=body, headers=headers)
        assert resp.status_code == 200
        assert hooks.operator_messages == []

    async def test_unknown_thread_logs_and_skips(
        self,
        stack: tuple[AsyncClient, _RecordingHooks],
    ) -> None:
        c, hooks = stack
        event = {
            "type": "event_callback",
            "event": {
                "type": "message",
                "thread_ts": "9999.9",
                "text": "from operator",
                "ts": "1700.9",
            },
        }
        body, headers = _signed(json.dumps(event).encode())
        resp = await c.post("/operator/slack/events", content=body, headers=headers)
        assert resp.status_code == 200
        assert hooks.operator_messages == []

    async def test_bad_signature_rejected(
        self,
        stack: tuple[AsyncClient, _RecordingHooks],
    ) -> None:
        c, _ = stack
        ts = str(int(time.time()))
        resp = await c.post(
            "/operator/slack/events",
            content=b'{"type":"url_verification","challenge":"x"}',
            headers={
                "x-slack-request-timestamp": ts,
                "x-slack-signature": "v0=" + "0" * 64,
                "content-type": "application/json",
            },
        )
        assert resp.status_code == 401

    async def test_malformed_json_rejected(
        self,
        stack: tuple[AsyncClient, _RecordingHooks],
    ) -> None:
        c, _ = stack
        body, headers = _signed(b"not json")
        resp = await c.post("/operator/slack/events", content=body, headers=headers)
        assert resp.status_code == 400


class TestInteractivity:
    async def test_resume_button_invokes_callback(
        self,
        stack: tuple[AsyncClient, _RecordingHooks],
    ) -> None:
        c, hooks = stack
        payload = {
            "type": "block_actions",
            "actions": [
                {"action_id": "resume_session", "value": "sess-Y"},
            ],
        }
        form_body = urlencode({"payload": json.dumps(payload)}).encode()
        ts = str(int(time.time()))
        sig = compute_signature(SECRET, ts, form_body)
        resp = await c.post(
            "/operator/slack/interactivity",
            content=form_body,
            headers={
                "x-slack-request-timestamp": ts,
                "x-slack-signature": sig,
                "content-type": "application/x-www-form-urlencoded",
            },
        )
        assert resp.status_code == 200
        assert hooks.resumed == ["sess-Y"]

    async def test_unknown_action_id_skipped(
        self,
        stack: tuple[AsyncClient, _RecordingHooks],
    ) -> None:
        c, hooks = stack
        payload = {
            "type": "block_actions",
            "actions": [{"action_id": "some_other_action", "value": "x"}],
        }
        form_body = urlencode({"payload": json.dumps(payload)}).encode()
        ts = str(int(time.time()))
        sig = compute_signature(SECRET, ts, form_body)
        resp = await c.post(
            "/operator/slack/interactivity",
            content=form_body,
            headers={
                "x-slack-request-timestamp": ts,
                "x-slack-signature": sig,
                "content-type": "application/x-www-form-urlencoded",
            },
        )
        assert resp.status_code == 200
        assert hooks.resumed == []

    async def test_bad_signature_rejected(
        self,
        stack: tuple[AsyncClient, _RecordingHooks],
    ) -> None:
        c, _ = stack
        form_body = urlencode({"payload": "{}"}).encode()
        resp = await c.post(
            "/operator/slack/interactivity",
            content=form_body,
            headers={
                "x-slack-request-timestamp": str(int(time.time())),
                "x-slack-signature": "v0=" + "0" * 64,
                "content-type": "application/x-www-form-urlencoded",
            },
        )
        assert resp.status_code == 401

    async def test_malformed_payload_rejected(
        self,
        stack: tuple[AsyncClient, _RecordingHooks],
    ) -> None:
        c, _ = stack
        form_body = urlencode({"payload": "not json"}).encode()
        ts = str(int(time.time()))
        sig = compute_signature(SECRET, ts, form_body)
        resp = await c.post(
            "/operator/slack/interactivity",
            content=form_body,
            headers={
                "x-slack-request-timestamp": ts,
                "x-slack-signature": sig,
                "content-type": "application/x-www-form-urlencoded",
            },
        )
        assert resp.status_code == 400
