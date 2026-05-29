"""Unit tests for ``backend.app.channels.wecom_aibot.client``.

The WebSocket transport itself is mocked (we don't stand up a real WS
server). These tests cover the pure-logic parts:

- inbound JSON frame normalization → :class:`SystemMessage`
- ``ping`` reply, ``pong``/``ack`` swallow, unknown-type drop, bad-JSON drop
- ``send_text`` no-op when disconnected; correct frame when connected
- ``stop`` flips the run-loop event and tolerates a missing socket
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from backend.app.channels.wecom_aibot.client import WecomAibotClient


def _make_client(debouncer: AsyncMock | MagicMock | None = None) -> WecomAibotClient:
    return WecomAibotClient(
        ws_url="ws://example/ws",
        token="tok",
        heartbeat_seconds=30,
        debouncer=debouncer or AsyncMock(),
    )


class TestHandleFrame:
    async def test_message_frame_dispatches_to_debouncer(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)
        await c._handle_frame({"type": "message", "from": "u1", "text": "hi", "msg_id": "m-1"})
        deb.observe.assert_awaited_once()
        sys_msg = deb.observe.await_args.args[0]
        assert sys_msg.channel == "wecom_aibot"
        assert sys_msg.channel_user_id == "u1"
        assert sys_msg.text == "hi"
        assert sys_msg.dedup_key == "m-1"

    async def test_message_without_msg_id_synthesizes_dedup_key(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)
        await c._handle_frame({"type": "message", "from": "u2", "text": "x"})
        sys_msg = deb.observe.await_args.args[0]
        assert sys_msg.dedup_key.startswith("wecom_aibot-u2-")

    async def test_message_missing_from_is_dropped(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)
        await c._handle_frame({"type": "message", "text": "no sender"})
        deb.observe.assert_not_called()

    async def test_message_missing_text_is_dropped(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)
        await c._handle_frame({"type": "message", "from": "u3"})
        deb.observe.assert_not_called()

    async def test_pong_and_ack_are_swallowed(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)
        await c._handle_frame({"type": "pong"})
        await c._handle_frame({"type": "ack"})
        deb.observe.assert_not_called()

    async def test_unknown_type_is_dropped(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)
        await c._handle_frame({"type": "weather"})
        deb.observe.assert_not_called()

    async def test_ping_replies_pong_when_connected(self) -> None:
        c = _make_client()
        ws = AsyncMock()
        c._ws = ws
        await c._handle_frame({"type": "ping"})
        ws.send.assert_awaited_once()
        sent = json.loads(ws.send.await_args.args[0])
        assert sent["type"] == "pong"
        assert "ts" in sent

    async def test_ping_when_disconnected_is_noop(self) -> None:
        c = _make_client()
        c._ws = None
        # Must not raise.
        await c._handle_frame({"type": "ping"})


class TestSendText:
    async def test_no_socket_drops_silently(self) -> None:
        c = _make_client()
        c._ws = None
        await c.send_text("u", "hi")  # must not raise

    async def test_sends_message_frame_when_connected(self) -> None:
        c = _make_client()
        ws = AsyncMock()
        c._ws = ws
        await c.send_text("u-7", "你好")
        ws.send.assert_awaited_once()
        frame = json.loads(ws.send.await_args.args[0])
        assert frame == {"type": "message", "to": "u-7", "text": "你好"}


class TestStop:
    async def test_stop_sets_event_and_closes_socket(self) -> None:
        c = _make_client()
        ws = AsyncMock()
        c._ws = ws
        await c.stop()
        assert c._stop.is_set()
        ws.close.assert_awaited_once()

    async def test_stop_with_no_socket_is_safe(self) -> None:
        c = _make_client()
        c._ws = None
        await c.stop()
        assert c._stop.is_set()

    async def test_stop_swallows_close_errors(self) -> None:
        c = _make_client()
        ws = AsyncMock()
        ws.close = AsyncMock(side_effect=RuntimeError("already gone"))
        c._ws = ws
        # Must not raise — shutdown path swallows the close error.
        await c.stop()
        assert c._stop.is_set()


class TestReadLoop:
    async def test_read_loop_dispatches_each_frame(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)

        class _StubWs:
            def __aiter__(self):
                async def gen():
                    yield json.dumps({"type": "message", "from": "u", "text": "a"})
                    yield json.dumps({"type": "message", "from": "u", "text": "b"})

                return gen()

        c._ws = _StubWs()  # type: ignore[assignment]
        await c._read_loop()
        assert deb.observe.await_count == 2

    async def test_read_loop_swallows_bad_json(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)

        class _StubWs:
            def __aiter__(self):
                async def gen():
                    yield "not-json"
                    yield json.dumps({"type": "message", "from": "u", "text": "ok"})

                return gen()

        c._ws = _StubWs()  # type: ignore[assignment]
        await c._read_loop()
        # Only the valid frame should reach the debouncer.
        deb.observe.assert_awaited_once()
