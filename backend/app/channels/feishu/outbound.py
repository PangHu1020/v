"""Feishu outbound message delivery.

Sends text replies via the Feishu Open API. The tenant access token is
fetched on demand and cached in-process for slightly less than its TTL.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx

from backend.v.utils.logging import get_logger

_log = get_logger("channels.feishu.outbound")

_BASE = "https://open.feishu.cn/open-apis"


class FeishuOutboundError(Exception):
    """Raised when the Feishu server rejects a token fetch or send."""


class FeishuOutbound:
    """In-process token-cached client for Feishu IM ``messages.create``."""

    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        client: httpx.AsyncClient | None = None,
        token_safety_margin_seconds: int = 60,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self._safety = token_safety_margin_seconds
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._token_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _tenant_access_token(self) -> str:
        async with self._token_lock:
            now = time.monotonic()
            if self._token and now < self._token_expires_at:
                return self._token

            resp = await self._client.post(
                f"{_BASE}/auth/v3/tenant_access_token/internal",
                json={"app_id": self._app_id, "app_secret": self._app_secret},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                raise FeishuOutboundError(f"tenant_access_token failed: {data}")

            self._token = data["tenant_access_token"]
            self._token_expires_at = now + max(0, int(data["expire"]) - self._safety)
            return self._token

    async def send_text(self, channel_user_id: str, text: str) -> None:
        """Deliver a text message to the customer.

        Args:
            channel_user_id: Feishu ``open_id`` for the recipient.
            text: Plain-text message content.
        """
        token = await self._tenant_access_token()
        resp = await self._client.post(
            f"{_BASE}/im/v1/messages",
            params={"receive_id_type": "open_id"},
            headers={"Authorization": f"Bearer {token}"},
            json={
                "receive_id": channel_user_id,
                "msg_type": "text",
                "content": json.dumps({"text": text}, ensure_ascii=False),
            },
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise FeishuOutboundError(f"messages.create failed: {data}")
        _log.info("feishu.outbound.sent", channel_user_id=channel_user_id, len=len(text))
