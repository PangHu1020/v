"""Unit tests for ``backend.app.channels.wecom_aibot.client``.

The WebSocket transport itself is mocked (we don't stand up a real WS
server). These tests cover the cmd-keyed wire protocol:

- inbound ``aibot_msg_callback`` → :class:`SystemMessage` (text + mixed)
- ``ping`` reply, unknown-cmd drop, bad-JSON drop
- subscribe handshake fires ``aibot_subscribe`` with bot_id + secret
- ``send_text`` issues ``aibot_respond_msg`` when a req_id is cached,
  else ``aibot_send_msg``; drops silently when disconnected
- ``stop`` flips the run-loop event and tolerates a missing socket
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from backend.app.channels.wecom_aibot.client import WecomAibotClient


def _make_client(debouncer: AsyncMock | MagicMock | None = None) -> WecomAibotClient:
    return WecomAibotClient(
        ws_url="ws://example/ws",
        bot_id="bot-1",
        secret="sec-1",
        heartbeat_seconds=30,
        debouncer=debouncer or AsyncMock(),
    )


def _callback_frame(
    *,
    msg_id: str = "m-1",
    chat_id: str = "chat-1",
    user_id: str = "u-1",
    msgtype: str = "text",
    content: str = "hi",
    req_id: str | None = "r-1",
) -> dict:
    body: dict = {
        "msgid": msg_id,
        "from": {"userid": user_id},
        "chatid": chat_id,
        "msgtype": msgtype,
    }
    if msgtype == "text":
        body["text"] = {"content": content}
    elif msgtype == "mixed":
        body["mixed"] = {
            "items": [
                {"type": "text", "text": {"content": content}},
            ]
        }
    frame: dict = {"cmd": "aibot_msg_callback", "body": body}
    if req_id:
        frame["header"] = {"req_id": req_id}
    return frame


class TestSubscribe:
    async def test_subscribe_sends_aibot_subscribe_payload(self) -> None:
        c = _make_client()
        ws = AsyncMock()
        c._ws = ws
        await c._subscribe()
        ws.send.assert_awaited_once()
        sent = json.loads(ws.send.await_args.args[0])
        assert sent == {
            "cmd": "aibot_subscribe",
            "body": {"bot_id": "bot-1", "secret": "sec-1"},
        }


class TestHandleFrame:
    async def test_text_callback_dispatches_to_debouncer(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)
        await c._handle_frame(_callback_frame(msg_id="m-7", chat_id="chat-7", content="hello"))
        deb.observe.assert_awaited_once()
        sys_msg = deb.observe.await_args.args[0]
        assert sys_msg.channel == "wecom_aibot"
        assert sys_msg.channel_user_id == "chat-7"
        assert sys_msg.text == "hello"
        assert sys_msg.dedup_key == "m-7"

    async def test_mixed_callback_concatenates_text_items(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)
        frame = {
            "cmd": "aibot_msg_callback",
            "header": {"req_id": "r-2"},
            "body": {
                "msgid": "m-2",
                "from": {"userid": "u"},
                "chatid": "c",
                "msgtype": "mixed",
                "mixed": {
                    "items": [
                        {"type": "text", "text": {"content": "part1"}},
                        {"type": "image", "image": {"url": "x"}},
                        {"type": "text", "text": {"content": "part2"}},
                    ]
                },
            },
        }
        await c._handle_frame(frame)
        deb.observe.assert_awaited_once()
        assert deb.observe.await_args.args[0].text == "part1 part2"

    async def test_chat_id_falls_back_to_user_id(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)
        frame = _callback_frame(chat_id="", user_id="u-fb")
        # Manually clear chatid since the helper sets it to "" but our
        # extractor falls back to userid in that case.
        frame["body"]["chatid"] = ""
        await c._handle_frame(frame)
        deb.observe.assert_awaited_once()
        assert deb.observe.await_args.args[0].channel_user_id == "u-fb"

    async def test_callback_caches_req_id_for_chat(self) -> None:
        c = _make_client()
        await c._handle_frame(_callback_frame(chat_id="c-1", req_id="REQ-1"))
        assert c._last_req_ids["c-1"] == "REQ-1"

    async def test_image_only_callback_is_dropped(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)
        frame = {
            "cmd": "aibot_msg_callback",
            "body": {
                "msgid": "m",
                "from": {"userid": "u"},
                "chatid": "c",
                "msgtype": "image",
                "image": {"url": "x"},
            },
        }
        await c._handle_frame(frame)
        # Empty extracted text → callback is observed only if text exists.
        deb.observe.assert_not_called()

    async def test_callback_without_chat_or_user_is_dropped(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)
        frame = {
            "cmd": "aibot_msg_callback",
            "body": {"msgid": "m", "from": {}, "msgtype": "text", "text": {"content": "x"}},
        }
        await c._handle_frame(frame)
        deb.observe.assert_not_called()

    async def test_unknown_cmd_is_dropped(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)
        await c._handle_frame({"cmd": "aibot_subscribe_ack"})
        deb.observe.assert_not_called()

    async def test_ping_replies_ping_when_connected(self) -> None:
        c = _make_client()
        ws = AsyncMock()
        c._ws = ws
        await c._handle_frame({"cmd": "ping"})
        ws.send.assert_awaited_once()
        sent = json.loads(ws.send.await_args.args[0])
        assert sent["cmd"] == "ping"
        assert "ts" in sent

    async def test_ping_when_disconnected_is_noop(self) -> None:
        c = _make_client()
        c._ws = None
        # Must not raise.
        await c._handle_frame({"cmd": "ping"})

    async def test_callback_synthesizes_msg_id_when_missing(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)
        frame = _callback_frame()
        frame["body"]["msgid"] = ""
        await c._handle_frame(frame)
        deb.observe.assert_awaited_once()
        assert deb.observe.await_args.args[0].dedup_key.startswith("wecom_aibot-chat-1-")

    async def test_callback_with_non_dict_body_is_skipped(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)
        await c._handle_frame({"cmd": "aibot_msg_callback", "body": "oops"})
        deb.observe.assert_not_called()


class TestSendText:
    async def test_no_socket_drops_silently(self) -> None:
        c = _make_client()
        c._ws = None
        await c.send_text("u", "hi")  # must not raise

    async def test_sends_aibot_send_msg_when_no_req_id(self) -> None:
        c = _make_client()
        ws = AsyncMock()
        c._ws = ws
        await c.send_text("chat-x", "你好")
        ws.send.assert_awaited_once()
        sent = json.loads(ws.send.await_args.args[0])
        assert sent["cmd"] == "aibot_send_msg"
        assert sent["body"]["chatid"] == "chat-x"
        assert sent["body"]["msgtype"] == "markdown"
        assert sent["body"]["markdown"]["content"] == "你好"

    async def test_sends_aibot_respond_msg_when_req_id_cached(self) -> None:
        c = _make_client()
        ws = AsyncMock()
        c._ws = ws
        c._last_req_ids["chat-y"] = "REQ-42"
        await c.send_text("chat-y", "回复")
        sent = json.loads(ws.send.await_args.args[0])
        assert sent["cmd"] == "aibot_respond_msg"
        assert sent["header"] == {"req_id": "REQ-42"}
        assert sent["body"]["msgtype"] == "markdown"
        assert sent["body"]["markdown"]["content"] == "回复"
        # respond_msg shouldn't carry chatid — req_id does the routing.
        assert "chatid" not in sent["body"]

    async def test_truncates_to_max_message_length(self) -> None:
        c = _make_client()
        ws = AsyncMock()
        c._ws = ws
        long_text = "a" * 5000
        await c.send_text("chat-z", long_text)
        sent = json.loads(ws.send.await_args.args[0])
        assert len(sent["body"]["markdown"]["content"]) == 4000


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
    async def test_read_loop_dispatches_each_callback(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)

        class _StubWs:
            def __aiter__(self):
                async def gen():
                    yield json.dumps(_callback_frame(msg_id="m-a"))
                    yield json.dumps(_callback_frame(msg_id="m-b"))

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
                    yield json.dumps(_callback_frame())

                return gen()

        c._ws = _StubWs()  # type: ignore[assignment]
        await c._read_loop()
        # Only the valid callback should reach the debouncer.
        deb.observe.assert_awaited_once()

    async def test_read_loop_skips_non_dict_frames(self) -> None:
        deb = AsyncMock()
        c = _make_client(deb)

        class _StubWs:
            def __aiter__(self):
                async def gen():
                    yield json.dumps([1, 2, 3])  # array, not dict

                return gen()

        c._ws = _StubWs()  # type: ignore[assignment]
        await c._read_loop()
        deb.observe.assert_not_called()
