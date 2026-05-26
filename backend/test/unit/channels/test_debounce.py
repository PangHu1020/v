"""Unit tests for ``backend.app.channels.debounce``.

Validates that bursts of messages from the same identity collapse into one
dispatch, that distinct identities are independent, and that the merged
text + dedup_key are concatenated in arrival order.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest

from backend.app.bus.messages import SystemMessage
from backend.app.channels.debounce import Debouncer


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


def _msg(
    *,
    user: str = "ext-1",
    text: str,
    dedup: str | None = None,
    channel: str = "wecom",
) -> SystemMessage:
    return SystemMessage(
        channel=channel,  # type: ignore[arg-type]
        channel_user_id=user,
        text=text,
        dedup_key=dedup or f"d-{user}-{text}",
    )


class TestDebouncer:
    async def test_single_message_dispatched_after_window(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        seen: list[SystemMessage] = []

        async def dispatch(m: SystemMessage) -> None:
            seen.append(m)

        deb = Debouncer(redis_client, window_ms=80, dispatch=dispatch)
        await deb.observe(_msg(text="hello"))
        await asyncio.sleep(0.18)

        assert len(seen) == 1
        assert seen[0].text == "hello"
        await deb.shutdown()

    async def test_burst_collapses_into_one_dispatch(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        seen: list[SystemMessage] = []

        async def dispatch(m: SystemMessage) -> None:
            seen.append(m)

        deb = Debouncer(redis_client, window_ms=120, dispatch=dispatch)
        await deb.observe(_msg(text="hi"))
        await asyncio.sleep(0.04)
        await deb.observe(_msg(text="i have"))
        await asyncio.sleep(0.04)
        await deb.observe(_msg(text="a question"))
        await asyncio.sleep(0.30)

        assert len(seen) == 1
        # Merged text preserves arrival order joined by newlines.
        assert seen[0].text == "hi\ni have\na question"
        # Dedup key concatenates all original ids.
        assert seen[0].dedup_key.count("|") == 2
        await deb.shutdown()

    async def test_distinct_users_are_independent(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        seen: list[SystemMessage] = []

        async def dispatch(m: SystemMessage) -> None:
            seen.append(m)

        deb = Debouncer(redis_client, window_ms=80, dispatch=dispatch)
        await deb.observe(_msg(user="alice", text="hi from alice"))
        await deb.observe(_msg(user="bob", text="hi from bob"))
        await asyncio.sleep(0.20)

        assert len(seen) == 2
        users = {m.channel_user_id for m in seen}
        assert users == {"alice", "bob"}
        await deb.shutdown()

    async def test_late_arrival_resets_timer(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        # If a second message arrives before the window expires, the flush
        # is rescheduled and we still get a single merged dispatch.
        seen: list[SystemMessage] = []

        async def dispatch(m: SystemMessage) -> None:
            seen.append(m)

        deb = Debouncer(redis_client, window_ms=100, dispatch=dispatch)
        await deb.observe(_msg(text="a"))
        await asyncio.sleep(0.06)
        await deb.observe(_msg(text="b"))
        await asyncio.sleep(0.06)
        # By now we are 120ms past the first message; without timer reset
        # the dispatcher would have already fired with just "a".
        await deb.observe(_msg(text="c"))
        await asyncio.sleep(0.20)

        assert len(seen) == 1
        assert seen[0].text == "a\nb\nc"
        await deb.shutdown()

    async def test_shutdown_cancels_pending(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        seen: list[SystemMessage] = []

        async def dispatch(m: SystemMessage) -> None:
            seen.append(m)

        deb = Debouncer(redis_client, window_ms=200, dispatch=dispatch)
        await deb.observe(_msg(text="never delivered"))
        await deb.shutdown()
        await asyncio.sleep(0.30)

        assert len(seen) == 0
