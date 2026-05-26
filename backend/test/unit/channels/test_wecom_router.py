"""Unit tests for ``backend.app.channels.wecom.router``.

Build a tiny FastAPI app, mount the WeCom router, and exercise both the
URL-verification echo and the encrypted-event POST path. The debouncer is
stubbed; what we verify here is that the router parses, validates, and
hands a properly normalized :class:`SystemMessage` to ``debouncer.observe``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.app.bus.messages import SystemMessage
from backend.app.channels.wecom.crypto import WecomCrypto
from backend.app.channels.wecom.router import build_router
from backend.app.channels.wecom.signature import compute_signature
from backend.app.gateway.middleware import RequestIdMiddleware
from backend.v.configs.base import WecomSettings

TEST_AES_KEY = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ"
TEST_CORP_ID = "wxabcdef1234567890"
TEST_TOKEN = "qwerty"
TEST_AGENT_ID = "1000002"


class _StubDebouncer:
    """Captures observed messages without scheduling a real flush."""

    def __init__(self) -> None:
        self.observed: list[SystemMessage] = []

    async def observe(self, message: SystemMessage) -> None:
        self.observed.append(message)

    async def shutdown(self) -> None:
        pass


@pytest.fixture
def settings() -> WecomSettings:
    return WecomSettings(
        _env_file=None,  # type: ignore[call-arg]
        corp_id=TEST_CORP_ID,
        agent_id=TEST_AGENT_ID,
        secret="s",
        token=TEST_TOKEN,
        aes_key=TEST_AES_KEY,
    )


@pytest.fixture
def crypto() -> WecomCrypto:
    return WecomCrypto(TEST_AES_KEY, TEST_CORP_ID)


@pytest.fixture
def debouncer() -> _StubDebouncer:
    return _StubDebouncer()


@pytest.fixture
async def client(
    settings: WecomSettings,
    crypto: WecomCrypto,
    debouncer: _StubDebouncer,
) -> AsyncIterator[tuple[AsyncClient, _StubDebouncer]]:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    app.include_router(build_router(settings, crypto, debouncer))  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c, debouncer


def _xml_with_encrypt(encrypt_b64: str) -> str:
    return (
        "<xml>"
        "<ToUserName><![CDATA[corp]]></ToUserName>"
        f"<Encrypt><![CDATA[{encrypt_b64}]]></Encrypt>"
        "</xml>"
    )


def _inner_text_xml(from_user: str, content: str, msg_id: str = "1234567") -> str:
    return (
        "<xml>"
        "<ToUserName><![CDATA[corp]]></ToUserName>"
        f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
        "<CreateTime>1700000000</CreateTime>"
        "<MsgType><![CDATA[text]]></MsgType>"
        f"<Content><![CDATA[{content}]]></Content>"
        f"<MsgId>{msg_id}</MsgId>"
        f"<AgentID>{TEST_AGENT_ID}</AgentID>"
        "</xml>"
    )


class TestUrlVerification:
    async def test_returns_decrypted_echostr(
        self,
        client: tuple[AsyncClient, _StubDebouncer],
        crypto: WecomCrypto,
    ) -> None:
        c, _ = client
        plaintext = "hello-echo"
        echostr = crypto.encrypt(plaintext)
        sig = compute_signature(TEST_TOKEN, "ts", "n", echostr)

        resp = await c.get(
            "/webhook/wecom",
            params={
                "msg_signature": sig,
                "timestamp": "ts",
                "nonce": "n",
                "echostr": echostr,
            },
        )
        assert resp.status_code == 200
        assert resp.text == plaintext

    async def test_bad_signature_rejected(
        self, client: tuple[AsyncClient, _StubDebouncer], crypto: WecomCrypto
    ) -> None:
        c, _ = client
        echostr = crypto.encrypt("plaintext")
        resp = await c.get(
            "/webhook/wecom",
            params={
                "msg_signature": "0" * 40,
                "timestamp": "ts",
                "nonce": "n",
                "echostr": echostr,
            },
        )
        assert resp.status_code == 401


class TestReceiveEvent:
    async def test_valid_text_event_observed(
        self,
        client: tuple[AsyncClient, _StubDebouncer],
        crypto: WecomCrypto,
    ) -> None:
        c, debouncer = client
        inner = _inner_text_xml("ext-42", "hello world", msg_id="m-1")
        encrypted = crypto.encrypt(inner)
        envelope = _xml_with_encrypt(encrypted)
        sig = compute_signature(TEST_TOKEN, "ts", "n", encrypted)

        resp = await c.post(
            "/webhook/wecom",
            params={"msg_signature": sig, "timestamp": "ts", "nonce": "n"},
            content=envelope,
        )
        assert resp.status_code == 200
        assert resp.text == "success"
        assert len(debouncer.observed) == 1
        msg = debouncer.observed[0]
        assert msg.channel == "wecom"
        assert msg.channel_user_id == "ext-42"
        assert msg.text == "hello world"
        assert msg.dedup_key == "m-1"

    async def test_bad_signature_rejected(
        self,
        client: tuple[AsyncClient, _StubDebouncer],
        crypto: WecomCrypto,
    ) -> None:
        c, debouncer = client
        encrypted = crypto.encrypt(_inner_text_xml("u", "hi"))
        envelope = _xml_with_encrypt(encrypted)
        resp = await c.post(
            "/webhook/wecom",
            params={"msg_signature": "0" * 40, "timestamp": "ts", "nonce": "n"},
            content=envelope,
        )
        assert resp.status_code == 401
        assert debouncer.observed == []

    async def test_non_text_messages_skipped(
        self,
        client: tuple[AsyncClient, _StubDebouncer],
        crypto: WecomCrypto,
    ) -> None:
        c, debouncer = client
        inner = (
            "<xml>"
            "<FromUserName><![CDATA[u]]></FromUserName>"
            "<MsgType><![CDATA[image]]></MsgType>"
            "<MediaId><![CDATA[m1]]></MediaId>"
            "<MsgId>2</MsgId>"
            "</xml>"
        )
        encrypted = crypto.encrypt(inner)
        envelope = _xml_with_encrypt(encrypted)
        sig = compute_signature(TEST_TOKEN, "ts", "n", encrypted)
        resp = await c.post(
            "/webhook/wecom",
            params={"msg_signature": sig, "timestamp": "ts", "nonce": "n"},
            content=envelope,
        )
        assert resp.status_code == 200
        assert debouncer.observed == []

    async def test_malformed_envelope_rejected(
        self,
        client: tuple[AsyncClient, _StubDebouncer],
    ) -> None:
        c, _ = client
        # Body is not even XML.
        resp = await c.post(
            "/webhook/wecom",
            params={"msg_signature": "x", "timestamp": "ts", "nonce": "n"},
            content="not xml",
        )
        assert resp.status_code == 400


class TestRouterWiring:
    async def test_response_has_request_id(
        self,
        client: tuple[AsyncClient, _StubDebouncer],
        crypto: WecomCrypto,
    ) -> None:
        c, _ = client
        plaintext = "x"
        echostr = crypto.encrypt(plaintext)
        sig = compute_signature(TEST_TOKEN, "ts", "n", echostr)
        resp = await c.get(
            "/webhook/wecom",
            params={
                "msg_signature": sig,
                "timestamp": "ts",
                "nonce": "n",
                "echostr": echostr,
            },
            headers={"x-request-id": "rid-1"},
        )
        assert resp.headers["x-request-id"] == "rid-1"


# Quiet ruff about the typing of the inner helper functions used above.
_ = Any
