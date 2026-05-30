"""Standalone worker process for the WeCom 智能机器人 WebSocket adapter.

Run as a separate process from the FastAPI app and the ARQ worker::

    uv run python -m backend.app.wecom_aibot_worker

Responsibilities (Phase-3 Group G):

- Open Redis + Postgres clients for this process.
- Construct the channel-layer :class:`Debouncer` and a
  :class:`BusProducer` so inbound WS frames feed the same bus the HTTP
  webhooks use.
- Hold a single persistent WebSocket via :class:`WecomAibotClient`,
  normalizing inbound frames to :class:`SystemMessage` and pushing them
  through the debouncer.
- Subscribe to the outbound Redis pub/sub channel and forward each
  published payload over the WS, providing a single-owner delivery
  point that other processes can publish to via
  :class:`WecomAibotOutbound`.
"""

from __future__ import annotations

import asyncio
import json
import signal

from backend.app.bus.messages import SystemMessage
from backend.app.bus.producer import BusProducer
from backend.app.bus.shard import RedisStreamShard
from backend.app.channels.debounce import Debouncer
from backend.app.channels.wecom_aibot.client import WecomAibotClient
from backend.app.store import close_client, create_client
from backend.v.configs import get_settings
from backend.v.utils.logging import configure as configure_logging
from backend.v.utils.logging import get_logger
from backend.v.utils.tracing import configure_langsmith

_log = get_logger("app.wecom_aibot_worker")


async def _outbound_subscriber(
    redis,
    *,
    pubsub_channel: str,
    client: WecomAibotClient,
    stop: asyncio.Event,
) -> None:
    """Forward every payload published on ``pubsub_channel`` over the WS."""
    pubsub = redis.pubsub()
    await pubsub.subscribe(pubsub_channel)
    _log.info("wecom_aibot_worker.outbound.subscribed", channel=pubsub_channel)
    try:
        while not stop.is_set():
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg is None:
                continue
            data = msg.get("data")
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            try:
                payload = json.loads(data)
            except (TypeError, json.JSONDecodeError) as exc:
                _log.warning("wecom_aibot_worker.outbound.bad_json", error=str(exc))
                continue
            channel_user_id = payload.get("channel_user_id")
            text = payload.get("text")
            if not channel_user_id or text is None:
                _log.warning("wecom_aibot_worker.outbound.missing_fields", payload=payload)
                continue
            try:
                await client.send_text(channel_user_id, text)
            except Exception as exc:
                _log.error("wecom_aibot_worker.outbound.send_failed", error=str(exc))
    finally:
        try:
            await pubsub.unsubscribe(pubsub_channel)
            await pubsub.aclose()
        except Exception:  # noqa: S110 — shutdown teardown, ignore failures
            pass


async def main() -> None:
    settings = get_settings()
    configure_logging(level=settings.runtime.log_level, json=settings.runtime.env != "dev")
    configure_langsmith(settings.langsmith)

    if not settings.wecom_aibot.ws_url:
        _log.error("wecom_aibot_worker.disabled", reason="WECOM_AIBOT_WS_URL is empty")
        return

    _log.info("wecom_aibot_worker.startup", env=settings.runtime.env)

    redis = await create_client(settings.redis.url)
    shard = RedisStreamShard(
        redis,
        prefix=settings.bus.stream_prefix,
        shard_count=settings.bus.shard_count,
    )
    producer = BusProducer(shard)

    async def channel_to_bus(message: SystemMessage) -> None:
        await producer.enqueue(message)

    debouncer = Debouncer(
        redis,
        window_ms=settings.bus.debounce_ms,
        dispatch=channel_to_bus,
    )

    client = WecomAibotClient(
        ws_url=settings.wecom_aibot.ws_url,
        bot_id=settings.wecom_aibot.bot_id,
        secret=settings.wecom_aibot.secret,
        heartbeat_seconds=settings.wecom_aibot.heartbeat_seconds,
        debouncer=debouncer,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass  # Windows / non-main thread fallback

    ws_task = asyncio.create_task(client.run())
    sub_task = asyncio.create_task(
        _outbound_subscriber(
            redis,
            pubsub_channel=settings.wecom_aibot.outbound_pubsub_channel,
            client=client,
            stop=stop,
        )
    )

    try:
        await stop.wait()
    finally:
        _log.info("wecom_aibot_worker.shutdown")
        await client.stop()
        await debouncer.shutdown()
        for t in (ws_task, sub_task):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: S110 — final shutdown
                pass
        await close_client(redis)


if __name__ == "__main__":
    asyncio.run(main())
