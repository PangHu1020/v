"""Slack outbound: post handoff alerts and relay messages.

Two key entry points:

- :meth:`SlackOutbound.post_handoff_alert` posts a rich-block message into
  the configured handoff channel summarizing the customer's conversation
  and offering a "Resume AI" button. Returns the Slack thread timestamp
  (``ts``) which becomes the operator-side anchor for that session.
- :meth:`SlackOutbound.post_customer_message` forwards continued customer
  messages (received during the suspension window) into the alert thread
  so the human operator sees them in real time.

A Redis mapping ``slack_thread:{ts} -> session_id`` survives process
restarts and is consulted by the inbound router (operator's reply -> which
session?).
"""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as redis_async
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from backend.v.utils.logging import get_logger

_log = get_logger("operator.slack.outbound")

_THREAD_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days; long enough for typical handoffs


class SlackOutboundError(Exception):
    """Raised when the Slack API rejects a call."""


def _alert_blocks(
    *,
    session_id: str,
    channel: str,
    channel_user_id: str,
    customer_text: str,
    transfer_reason: str,
) -> list[dict[str, Any]]:
    """Build Block Kit blocks for the handoff alert message."""
    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "🚨 人工接管告警"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*渠道:*\n{channel}"},
                {"type": "mrkdwn", "text": f"*客户:*\n{channel_user_id}"},
                {"type": "mrkdwn", "text": f"*会话:*\n`{session_id}`"},
                {"type": "mrkdwn", "text": f"*原因:*\n{transfer_reason}"},
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*最近的客户消息:*\n>>> {customer_text or '(无)'}",
            },
        },
        {
            "type": "actions",
            "block_id": "handoff_actions",
            "elements": [
                {
                    "type": "button",
                    "action_id": "resume_session",
                    "style": "primary",
                    "text": {"type": "plain_text", "text": "✅ Resume AI"},
                    "value": session_id,
                    "confirm": {
                        "title": {"type": "plain_text", "text": "确认继续 AI 接管"},
                        "text": {
                            "type": "plain_text",
                            "text": "这会让 AI 接着回复后续消息。",
                        },
                        "confirm": {"type": "plain_text", "text": "继续"},
                        "deny": {"type": "plain_text", "text": "取消"},
                    },
                },
            ],
        },
    ]


class SlackOutbound:
    """Async Slack Web API client scoped to the handoff alert channel."""

    def __init__(
        self,
        *,
        bot_token: str,
        alert_channel_id: str,
        redis: redis_async.Redis,
        client: AsyncWebClient | None = None,
    ) -> None:
        self._client = client or AsyncWebClient(token=bot_token)
        self._alert_channel = alert_channel_id
        self._redis = redis

    async def aclose(self) -> None:
        # AsyncWebClient owns an aiohttp session that must be closed cleanly.
        await self._client.close()

    async def post_handoff_alert(
        self,
        *,
        session_id: str,
        channel: str,
        channel_user_id: str,
        customer_text: str,
        transfer_reason: str,
    ) -> str:
        """Post the handoff alert and return the Slack thread ``ts``."""
        blocks = _alert_blocks(
            session_id=session_id,
            channel=channel,
            channel_user_id=channel_user_id,
            customer_text=customer_text,
            transfer_reason=transfer_reason,
        )
        try:
            resp = await self._client.chat_postMessage(
                channel=self._alert_channel,
                blocks=blocks,
                text=f"人工接管告警 ({channel}/{channel_user_id})",
            )
        except SlackApiError as exc:
            raise SlackOutboundError(f"chat.postMessage failed: {exc.response.data}") from exc

        thread_ts = resp["ts"]
        await self._redis.set(
            f"slack_thread:{thread_ts}",
            session_id,
            ex=_THREAD_TTL_SECONDS,
        )
        await self._redis.set(
            f"session_thread:{session_id}",
            json.dumps({"channel_id": self._alert_channel, "thread_ts": thread_ts}),
            ex=_THREAD_TTL_SECONDS,
        )
        _log.info(
            "operator.slack.alert_posted",
            session_id=session_id,
            thread_ts=thread_ts,
        )
        return thread_ts

    async def post_customer_message(
        self,
        *,
        thread_ts: str,
        text: str,
    ) -> None:
        """Forward a customer message into the alert thread for operator visibility."""
        try:
            await self._client.chat_postMessage(
                channel=self._alert_channel,
                thread_ts=thread_ts,
                text=f"📩 客户：{text}",
            )
        except SlackApiError as exc:
            raise SlackOutboundError(
                f"chat.postMessage (thread) failed: {exc.response.data}"
            ) from exc

    async def get_thread_for_session(self, session_id: str) -> str | None:
        """Return the Slack ``thread_ts`` that anchors ``session_id``, or ``None``."""
        raw = await self._redis.get(f"session_thread:{session_id}")
        if not raw:
            return None
        data = json.loads(raw)
        return data.get("thread_ts")
