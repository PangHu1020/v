"""Unit tests for ``backend.app.channels.feishu.router``."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.app.bus.messages import SystemMessage
from backend.app.channels.feishu.crypto import FeishuCrypto
from backend.app.channels.feishu.router import build_router
from backend.app.channels.feishu.signature import compute_signature
from backend.app.gateway.middleware import RequestIdMiddleware
from backend.v.configs.base import FeishuSettings

TEST_ENCRYPT_KEY = "feishu-test-key"


class _StubDebouncer:
    def __init__(self) -> None:
        self.observed: list[SystemMessage] = []

    async def observe(self, message: SystemMessage) -> None:
        self.observed.append(message)


@pytest.fixture
def settings() -> FeishuSettings:
    return FeishuSettings(
        _env_file=None,  # type: ignore[call-arg]
        app_id="cli_x",
        app_secret="s",
        encrypt_key=TEST_ENCRYPT_KEY,
        verification_token="v",
    )


@pytest.fixture
def crypto() -> FeishuCrypto:
    return FeishuCrypto(TEST_ENCRYPT_KEY)


@pytest.fixture
def debouncer() -> _StubDebouncer:
    return _StubDebouncer()


@pytest.fixture
async def client(
    settings: FeishuSettings,
    crypto: FeishuCrypto,
    debouncer: _StubDebouncer,
) -> AsyncIterator[tuple[AsyncClient, _StubDebouncer]]:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    app.include_router(build_router(settings, crypto, debouncer))  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c, debouncer


def _signed_post(
    crypto: FeishuCrypto,
    inner_payload: dict,
) -> tuple[bytes, dict[str, str]]:
    """Build an encrypted-event request body + signed headers."""
    encrypted = crypto.encrypt(json.dumps(inner_payload, ensure_ascii=False))
    body = json.dumps({"encrypt": encrypted}).encode("utf-8")
    sig = compute_signature("ts", "n", TEST_ENCRYPT_KEY, body)
    headers = {
        "x-lark-request-timestamp": "ts",
        "x-lark-request-nonce": "n",
        "x-lark-signature": sig,
        "content-type": "application/json",
    }
    return body, headers


class TestUrlVerification:
    async def test_outer_handshake(self, client: tuple[AsyncClient, _StubDebouncer]) -> None:
        c, _ = client
        body = json.dumps({"type": "url_verification", "challenge": "ch-1"}).encode("utf-8")
        # Signature optional for outer handshake (Feishu does not require
        # encryption for this case in some setups), but we still send one.
        sig = compute_signature("ts", "n", TEST_ENCRYPT_KEY, body)
        resp = await c.post(
            "/webhook/feishu",
            content=body,
            headers={
                "content-type": "application/json",
                "x-lark-request-timestamp": "ts",
                "x-lark-request-nonce": "n",
                "x-lark-signature": sig,
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {"challenge": "ch-1"}

    async def test_inner_handshake_in_encrypted_envelope(
        self,
        client: tuple[AsyncClient, _StubDebouncer],
        crypto: FeishuCrypto,
    ) -> None:
        c, _ = client
        inner = {"type": "url_verification", "challenge": "ch-encrypted"}
        body, headers = _signed_post(crypto, inner)
        resp = await c.post("/webhook/feishu", content=body, headers=headers)
        assert resp.status_code == 200
        assert resp.json() == {"challenge": "ch-encrypted"}


class TestReceiveEvent:
    async def test_valid_text_event_observed(
        self,
        client: tuple[AsyncClient, _StubDebouncer],
        crypto: FeishuCrypto,
    ) -> None:
        c, debouncer = client
        inner = {
            "schema": "2.0",
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_42"}},
                "message": {
                    "message_id": "om_xyz",
                    "message_type": "text",
                    "content": json.dumps({"text": "hi there"}, ensure_ascii=False),
                },
            },
        }
        body, headers = _signed_post(crypto, inner)
        resp = await c.post("/webhook/feishu", content=body, headers=headers)
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        assert len(debouncer.observed) == 1
        msg = debouncer.observed[0]
        assert msg.channel == "feishu"
        assert msg.channel_user_id == "ou_42"
        assert msg.text == "hi there"
        assert msg.dedup_key == "om_xyz"

    async def test_chinese_round_trip(
        self,
        client: tuple[AsyncClient, _StubDebouncer],
        crypto: FeishuCrypto,
    ) -> None:
        c, debouncer = client
        inner = {
            "event": {
                "sender": {"sender_id": {"open_id": "ou_zh"}},
                "message": {
                    "message_id": "om_zh",
                    "message_type": "text",
                    "content": json.dumps({"text": "请问订单状态"}, ensure_ascii=False),
                },
            }
        }
        body, headers = _signed_post(crypto, inner)
        resp = await c.post("/webhook/feishu", content=body, headers=headers)
        assert resp.status_code == 200
        assert debouncer.observed[0].text == "请问订单状态"

    async def test_bad_signature_rejected(
        self,
        client: tuple[AsyncClient, _StubDebouncer],
        crypto: FeishuCrypto,
    ) -> None:
        c, debouncer = client
        body, headers = _signed_post(crypto, {"event": {}})
        headers["x-lark-signature"] = "0" * 64
        resp = await c.post("/webhook/feishu", content=body, headers=headers)
        assert resp.status_code == 401
        assert debouncer.observed == []

    async def test_non_text_message_skipped(
        self,
        client: tuple[AsyncClient, _StubDebouncer],
        crypto: FeishuCrypto,
    ) -> None:
        c, debouncer = client
        inner = {
            "event": {
                "sender": {"sender_id": {"open_id": "ou_x"}},
                "message": {
                    "message_id": "om_img",
                    "message_type": "image",
                    "content": json.dumps({"image_key": "k"}),
                },
            }
        }
        body, headers = _signed_post(crypto, inner)
        resp = await c.post("/webhook/feishu", content=body, headers=headers)
        assert resp.status_code == 200
        assert debouncer.observed == []

    async def test_non_message_event_skipped(
        self,
        client: tuple[AsyncClient, _StubDebouncer],
        crypto: FeishuCrypto,
    ) -> None:
        # Member-added event has no sender.sender_id.open_id.
        c, debouncer = client
        inner = {"event": {"chat_id": "oc_x", "type": "added"}}
        body, headers = _signed_post(crypto, inner)
        resp = await c.post("/webhook/feishu", content=body, headers=headers)
        assert resp.status_code == 200
        assert debouncer.observed == []

    async def test_malformed_outer_json_rejected(
        self, client: tuple[AsyncClient, _StubDebouncer]
    ) -> None:
        c, _ = client
        resp = await c.post(
            "/webhook/feishu",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400


_ = Any
