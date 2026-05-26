"""Unit tests for ``backend.app.channels.wecom.outbound`` using respx."""

from __future__ import annotations

import httpx
import pytest
import respx

from backend.app.channels.wecom.outbound import WecomOutbound, WecomOutboundError


def _outbound() -> WecomOutbound:
    return WecomOutbound(
        corp_id="wxabc",
        secret="secret-x",
        agent_id="1000002",
        client=httpx.AsyncClient(timeout=5.0),
        token_safety_margin_seconds=1,
    )


class TestAccessToken:
    @respx.mock
    async def test_fetches_and_caches(self) -> None:
        out = _outbound()
        token_route = respx.get("https://qyapi.weixin.qq.com/cgi-bin/gettoken").mock(
            return_value=httpx.Response(
                200, json={"errcode": 0, "access_token": "tk-1", "expires_in": 7200}
            )
        )
        send_route = respx.post("https://qyapi.weixin.qq.com/cgi-bin/message/send").mock(
            return_value=httpx.Response(200, json={"errcode": 0})
        )

        await out.send_text("ext-1", "hi")
        await out.send_text("ext-2", "hi again")

        assert token_route.call_count == 1  # cached after first call
        assert send_route.call_count == 2
        await out.aclose()

    @respx.mock
    async def test_token_failure_raises(self) -> None:
        out = _outbound()
        respx.get("https://qyapi.weixin.qq.com/cgi-bin/gettoken").mock(
            return_value=httpx.Response(200, json={"errcode": 40001, "errmsg": "bad secret"})
        )
        with pytest.raises(WecomOutboundError, match="gettoken failed"):
            await out.send_text("ext-1", "hi")
        await out.aclose()


class TestSendText:
    @respx.mock
    async def test_sends_correct_payload(self) -> None:
        out = _outbound()
        respx.get("https://qyapi.weixin.qq.com/cgi-bin/gettoken").mock(
            return_value=httpx.Response(
                200, json={"errcode": 0, "access_token": "tk", "expires_in": 7200}
            )
        )
        send_route = respx.post("https://qyapi.weixin.qq.com/cgi-bin/message/send").mock(
            return_value=httpx.Response(200, json={"errcode": 0})
        )

        await out.send_text("ext-42", "hello world")

        assert send_route.called
        sent = send_route.calls.last.request
        body = sent.read()
        assert b"ext-42" in body
        assert b"hello world" in body
        assert b"1000002" in body  # agent_id
        await out.aclose()

    @respx.mock
    async def test_send_failure_raises(self) -> None:
        out = _outbound()
        respx.get("https://qyapi.weixin.qq.com/cgi-bin/gettoken").mock(
            return_value=httpx.Response(
                200, json={"errcode": 0, "access_token": "tk", "expires_in": 7200}
            )
        )
        respx.post("https://qyapi.weixin.qq.com/cgi-bin/message/send").mock(
            return_value=httpx.Response(200, json={"errcode": 81013, "errmsg": "user not in agent"})
        )
        with pytest.raises(WecomOutboundError, match="message/send failed"):
            await out.send_text("ext-x", "hi")
        await out.aclose()
