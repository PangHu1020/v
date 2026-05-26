"""Unit tests for ``backend.app.channels.feishu.outbound`` using respx."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from backend.app.channels.feishu.outbound import FeishuOutbound, FeishuOutboundError


def _outbound() -> FeishuOutbound:
    return FeishuOutbound(
        app_id="cli_x",
        app_secret="s",
        client=httpx.AsyncClient(timeout=5.0),
        token_safety_margin_seconds=1,
    )


class TestTenantToken:
    @respx.mock
    async def test_fetches_and_caches(self) -> None:
        out = _outbound()
        token_route = respx.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        ).mock(
            return_value=httpx.Response(
                200, json={"code": 0, "tenant_access_token": "tk-1", "expire": 7200}
            )
        )
        send_route = respx.post("https://open.feishu.cn/open-apis/im/v1/messages").mock(
            return_value=httpx.Response(200, json={"code": 0, "data": {}})
        )

        await out.send_text("ou_1", "hi")
        await out.send_text("ou_2", "hi again")

        assert token_route.call_count == 1
        assert send_route.call_count == 2
        await out.aclose()

    @respx.mock
    async def test_token_failure_raises(self) -> None:
        out = _outbound()
        respx.post("https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal").mock(
            return_value=httpx.Response(200, json={"code": 99991663, "msg": "bad app"})
        )
        with pytest.raises(FeishuOutboundError, match="tenant_access_token failed"):
            await out.send_text("ou_x", "hi")
        await out.aclose()


class TestSendText:
    @respx.mock
    async def test_sends_correct_payload(self) -> None:
        out = _outbound()
        respx.post("https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal").mock(
            return_value=httpx.Response(
                200, json={"code": 0, "tenant_access_token": "tk", "expire": 7200}
            )
        )
        send_route = respx.post("https://open.feishu.cn/open-apis/im/v1/messages").mock(
            return_value=httpx.Response(200, json={"code": 0, "data": {}})
        )

        await out.send_text("ou_42", "你好")

        assert send_route.called
        sent = send_route.calls.last.request
        body = json.loads(sent.read())
        assert body["receive_id"] == "ou_42"
        assert body["msg_type"] == "text"
        assert json.loads(body["content"]) == {"text": "你好"}
        assert sent.headers["authorization"].startswith("Bearer ")
        await out.aclose()

    @respx.mock
    async def test_send_failure_raises(self) -> None:
        out = _outbound()
        respx.post("https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal").mock(
            return_value=httpx.Response(
                200, json={"code": 0, "tenant_access_token": "tk", "expire": 7200}
            )
        )
        respx.post("https://open.feishu.cn/open-apis/im/v1/messages").mock(
            return_value=httpx.Response(200, json={"code": 230001, "msg": "user not in chat"})
        )
        with pytest.raises(FeishuOutboundError, match=r"messages\.create failed"):
            await out.send_text("ou_x", "hi")
        await out.aclose()
