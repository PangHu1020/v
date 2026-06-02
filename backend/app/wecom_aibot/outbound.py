"""WeCom 智能机器人 outbound publisher.

The WebSocket is owned by a single standalone worker process. To let any
other process (FastAPI app, ARQ worker, Slack handoff) deliver a reply,
we publish ``{"channel_user_id": ..., "text": ...}`` JSON onto a Redis
pub/sub channel. The worker subscribes to that channel and forwards each
payload over its WS via :meth:`WecomAibotClient.send_text`.

This class implements the same ``send_text(channel_user_id, text)``
interface as the other channel outbounds (Wecom, Feishu) so the bus
worker / handoff hooks can stay channel-agnostic.
"""

from __future__ import annotations

import json

import redis.asyncio as redis_async

from backend.v.utils.logging import get_logger

_log = get_logger("channels.wecom_aibot.outbound")


class WecomAibotOutbound:
    """Pub/sub publisher mirroring the channel-outbound interface."""

    def __init__(self, redis: redis_async.Redis, *, pubsub_channel: str) -> None:
        self._redis = redis
        self._channel = pubsub_channel

    async def send_text(self, channel_user_id: str, text: str) -> None:
        """Publish an outbound reply for the WS worker to forward."""
        payload = json.dumps(
            {"channel_user_id": channel_user_id, "text": text},
            ensure_ascii=False,
        )
        delivered = await self._redis.publish(self._channel, payload)
        _log.info(
            "wecom_aibot.outbound.published",
            channel_user_id=channel_user_id,
            len=len(text),
            subscribers=delivered,
        )
