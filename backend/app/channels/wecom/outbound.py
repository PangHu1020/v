"""WeCom outbound message delivery.

Sends text replies via the WeCom Server API. The access token is fetched on
demand and cached in-process for slightly less than its TTL.

For Phase-1 single-instance deployments the in-process cache is fine; a
multi-instance deployment would move the cache into Redis to avoid
duplicate token fetches.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from backend.v.utils.logging import get_logger

_log = get_logger("channels.wecom.outbound")

_BASE = "https://qyapi.weixin.qq.com"


class WecomOutboundError(Exception):
    """Raised when the WeCom server rejects a token fetch or send."""


class WecomOutbound:
    """In-process token-cached client for WeCom Server API ``message/send``."""

    def __init__(
        self,
        *,
        corp_id: str,
        secret: str,
        agent_id: str,
        client: httpx.AsyncClient | None = None,
        token_safety_margin_seconds: int = 60,
    ) -> None:
        self._corp_id = corp_id
        self._secret = secret
        self._agent_id = agent_id
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self._safety = token_safety_margin_seconds
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._token_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _access_token(self) -> str:
        """Return a valid access token, fetching a fresh one if needed."""
        async with self._token_lock:
            now = time.monotonic()
            if self._token and now < self._token_expires_at:
                return self._token

            resp = await self._client.get(
                f"{_BASE}/cgi-bin/gettoken",
                params={"corpid": self._corp_id, "corpsecret": self._secret},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("errcode") != 0:
                raise WecomOutboundError(f"gettoken failed: {data}")

            self._token = data["access_token"]
            self._token_expires_at = now + max(0, int(data["expires_in"]) - self._safety)
            return self._token

    async def send_text(self, channel_user_id: str, text: str) -> None:
        """Deliver a text message to the customer.

        Args:
            channel_user_id: WeCom external user id.
            text: Plain-text message content. Must be UTF-8.
        """
        token = await self._access_token()
        body = {
            "touser": channel_user_id,
            "msgtype": "text",
            "agentid": self._agent_id,
            "text": {"content": text},
        }
        resp = await self._client.post(
            f"{_BASE}/cgi-bin/message/send",
            params={"access_token": token},
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("errcode") != 0:
            raise WecomOutboundError(f"message/send failed: {data}")
        _log.info("wecom.outbound.sent", channel_user_id=channel_user_id, len=len(text))
