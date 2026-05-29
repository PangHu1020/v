"""Unit tests for ``backend.app.wecom_aibot_worker``.

Covers the parts that are unit-testable without standing up a real WS:

- The outbound subscriber loop forwards valid payloads, swallows bad
  JSON, and drops payloads with missing fields.
- ``main()`` short-circuits when ``WECOM_AIBOT_WS_URL`` is empty
  (the disabled-channel guard).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app import wecom_aibot_worker


class _StubPubSub:
    """Replay a fixed list of pub/sub messages then return ``None`` forever."""

    def __init__(self, messages: list[dict[str, Any] | None]) -> None:
        self._queue = list(messages)
        self.subscribed_to: str | None = None
        self.unsubscribed_from: str | None = None
        self.closed = False

    async def subscribe(self, channel: str) -> None:
        self.subscribed_to = channel

    async def unsubscribe(self, channel: str) -> None:
        self.unsubscribed_from = channel

    async def aclose(self) -> None:
        self.closed = True

    async def get_message(
        self, *, ignore_subscribe_messages: bool = True, timeout: float = 1.0
    ) -> dict[str, Any] | None:
        # Yield so any scheduled call_later(stop.set) can fire — otherwise
        # the worker spins on an empty queue and the test never exits.
        await asyncio.sleep(0)
        if self._queue:
            return self._queue.pop(0)
        return None


def _redis_with(messages: list[dict[str, Any] | None]) -> tuple[MagicMock, _StubPubSub]:
    pubsub = _StubPubSub(messages)
    redis = MagicMock()
    redis.pubsub = MagicMock(return_value=pubsub)
    return redis, pubsub


class TestOutboundSubscriber:
    async def test_forwards_valid_payload(self) -> None:
        client = MagicMock()
        client.send_text = AsyncMock()
        redis, pubsub = _redis_with(
            [
                {
                    "type": "message",
                    "data": json.dumps({"channel_user_id": "u-1", "text": "hi"}).encode(),
                },
            ]
        )
        stop = asyncio.Event()

        async def stopper() -> None:
            await asyncio.sleep(0.05)
            stop.set()

        await asyncio.gather(
            wecom_aibot_worker._outbound_subscriber(
                redis, pubsub_channel="ch", client=client, stop=stop
            ),
            stopper(),
        )

        client.send_text.assert_awaited_once_with("u-1", "hi")
        assert pubsub.subscribed_to == "ch"
        assert pubsub.closed is True

    async def test_decodes_str_data_path(self) -> None:
        client = MagicMock()
        client.send_text = AsyncMock()
        redis, _ = _redis_with(
            [
                {
                    "type": "message",
                    "data": json.dumps({"channel_user_id": "u", "text": "x"}),
                },
            ]
        )
        stop = asyncio.Event()
        asyncio.get_event_loop().call_later(0.05, stop.set)
        await wecom_aibot_worker._outbound_subscriber(
            redis, pubsub_channel="ch", client=client, stop=stop
        )
        client.send_text.assert_awaited_once_with("u", "x")

    async def test_bad_json_is_skipped(self) -> None:
        client = MagicMock()
        client.send_text = AsyncMock()
        redis, _ = _redis_with([{"type": "message", "data": b"not-json"}])
        stop = asyncio.Event()
        asyncio.get_event_loop().call_later(0.05, stop.set)
        await wecom_aibot_worker._outbound_subscriber(
            redis, pubsub_channel="ch", client=client, stop=stop
        )
        client.send_text.assert_not_called()

    async def test_missing_fields_is_skipped(self) -> None:
        client = MagicMock()
        client.send_text = AsyncMock()
        redis, _ = _redis_with(
            [
                {"type": "message", "data": json.dumps({"text": "no user"}).encode()},
                {
                    "type": "message",
                    "data": json.dumps({"channel_user_id": "u"}).encode(),
                },
            ]
        )
        stop = asyncio.Event()
        asyncio.get_event_loop().call_later(0.05, stop.set)
        await wecom_aibot_worker._outbound_subscriber(
            redis, pubsub_channel="ch", client=client, stop=stop
        )
        client.send_text.assert_not_called()

    async def test_send_failure_is_logged_not_raised(self) -> None:
        client = MagicMock()
        client.send_text = AsyncMock(side_effect=RuntimeError("ws gone"))
        redis, _ = _redis_with(
            [
                {
                    "type": "message",
                    "data": json.dumps({"channel_user_id": "u", "text": "x"}).encode(),
                },
            ]
        )
        stop = asyncio.Event()
        asyncio.get_event_loop().call_later(0.05, stop.set)
        # Must not propagate the RuntimeError.
        await wecom_aibot_worker._outbound_subscriber(
            redis, pubsub_channel="ch", client=client, stop=stop
        )
        client.send_text.assert_awaited_once()


class TestMain:
    async def test_main_exits_early_when_ws_url_is_empty(self) -> None:
        # Build a minimal fake settings tree with the WS URL blank.
        fake_settings = MagicMock()
        fake_settings.runtime.log_level = "INFO"
        fake_settings.runtime.env = "dev"
        fake_settings.wecom_aibot.ws_url = ""

        with (
            patch.object(wecom_aibot_worker, "get_settings", return_value=fake_settings),
            patch.object(wecom_aibot_worker, "configure_logging") as mock_cfg,
            patch.object(wecom_aibot_worker, "create_client") as mock_create,
        ):
            await wecom_aibot_worker.main()

        mock_cfg.assert_called_once()
        # Early-exit guard: no Redis client should be constructed.
        mock_create.assert_not_called()
