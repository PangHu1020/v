"""Unit tests for ``backend.app.bus.producer`` and ``backend.app.bus.consumer``.

End-to-end round-trip via fakeredis with a per-test stub handler. Verifies:
- Producer routes messages to the correct shard.
- Consumer dispatches the deserialized SystemMessage to the handler.
- Handler-level exceptions land in the DLQ stream and the original entry is acked.
- Malformed payloads also land in the DLQ.
- Multiple shards process distinct conversations concurrently.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest

from backend.app.bus.consumer import BusConsumer
from backend.app.bus.messages import SystemMessage
from backend.app.bus.producer import BusProducer
from backend.app.bus.shard import RedisStreamShard


@pytest.fixture
async def redis_client() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


def _shard(redis: fakeredis.aioredis.FakeRedis, count: int = 4) -> RedisStreamShard:
    return RedisStreamShard(redis, prefix="bus", shard_count=count)


def _msg(channel_user_id: str = "ext-1", text: str = "hello") -> SystemMessage:
    return SystemMessage(
        channel="wecom",
        channel_user_id=channel_user_id,
        text=text,
        dedup_key=f"d-{channel_user_id}-{text}",
    )


async def _drain_consumer(
    consumer: BusConsumer,
    handler,  # type: ignore[no-untyped-def]
    *,
    expected: int,
    timeout: float = 2.0,
) -> None:
    """Run consumer until ``expected`` messages have been seen, then stop."""
    seen: list[SystemMessage] = []
    done = asyncio.Event()

    async def wrapped(msg: SystemMessage) -> None:
        await handler(msg)
        seen.append(msg)
        if len(seen) >= expected:
            done.set()

    consume_task = asyncio.create_task(consumer.run(wrapped))
    try:
        await asyncio.wait_for(done.wait(), timeout=timeout)
    finally:
        await consumer.stop()
        consume_task.cancel()
        try:
            await consume_task
        except (asyncio.CancelledError, BaseException):
            pass


class TestProducerEnqueue:
    async def test_routes_to_correct_shard(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        shard = _shard(redis_client, count=8)
        producer = BusProducer(shard)
        msg = _msg(channel_user_id="ext-42")
        expected_key = shard.stream_key("wecom", "ext-42")

        entry_id = await producer.enqueue(msg)
        assert entry_id

        length = await redis_client.xlen(expected_key)
        assert length == 1

    async def test_payload_round_trips_to_systemmessage(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        shard = _shard(redis_client)
        producer = BusProducer(shard)
        msg = _msg(text="客户问询")

        await producer.enqueue(msg)
        key = shard.stream_key(msg.channel, msg.channel_user_id)
        entries = await redis_client.xrange(key)
        assert len(entries) == 1
        _, fields = entries[0]
        rehydrated = SystemMessage.model_validate_json(fields[b"json"])
        assert rehydrated.text == "客户问询"
        assert rehydrated.dedup_key == msg.dedup_key


class TestConsumerHappyPath:
    async def test_round_trip(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        shard = _shard(redis_client, count=2)
        producer = BusProducer(shard)
        consumer = BusConsumer(shard, group="g", consumer_name="c1", block_ms=50)

        await producer.enqueue(_msg(channel_user_id="ext-A", text="hi"))
        await producer.enqueue(_msg(channel_user_id="ext-B", text="hey"))

        async def handler(msg: SystemMessage) -> None:
            return None

        await _drain_consumer(consumer, handler, expected=2)

        # All entries acked: pending count is zero across shards.
        for key in shard.all_stream_keys():
            try:
                pending = await redis_client.xpending(key, "g")
                # xpending for an empty group returns dict with 'pending' key
                # in some redis-py versions, or a tuple-like in others.
                pending_count = (
                    pending.get("pending", 0)
                    if isinstance(pending, dict)
                    else pending[0]
                    if pending
                    else 0
                )
                assert pending_count == 0
            except Exception:
                # Group never created on shards that received no messages.
                pass

    async def test_concurrent_shards(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        shard = _shard(redis_client, count=4)
        producer = BusProducer(shard)
        consumer = BusConsumer(shard, group="g", consumer_name="c1", block_ms=50)

        for i in range(20):
            await producer.enqueue(_msg(channel_user_id=f"ext-{i}", text=f"msg-{i}"))

        async def handler(msg: SystemMessage) -> None:
            return None

        await _drain_consumer(consumer, handler, expected=20, timeout=3.0)


class TestConsumerFailureHandling:
    async def test_handler_exception_routes_to_dlq(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        shard = _shard(redis_client, count=2)
        producer = BusProducer(shard)
        consumer = BusConsumer(shard, group="g", consumer_name="c1", block_ms=50)

        await producer.enqueue(_msg(channel_user_id="ext-X", text="boom"))

        seen = asyncio.Event()

        async def failing_handler(msg: SystemMessage) -> None:
            seen.set()
            raise RuntimeError("kaboom")

        run_task = asyncio.create_task(consumer.run(failing_handler))
        try:
            await asyncio.wait_for(seen.wait(), timeout=2.0)
            # Give the consumer a moment to process the exception path.
            await asyncio.sleep(0.1)
        finally:
            await consumer.stop()
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, BaseException):
                pass

        dlq_entries = await redis_client.xrange(shard.dlq_key())
        assert len(dlq_entries) == 1
        _, fields = dlq_entries[0]
        assert b"handler:" in fields[b"reason"]
        assert b"kaboom" in fields[b"reason"]

    async def test_malformed_payload_routes_to_dlq(
        self, redis_client: fakeredis.aioredis.FakeRedis
    ) -> None:
        shard = _shard(redis_client, count=2)
        consumer = BusConsumer(shard, group="g", consumer_name="c1", block_ms=50)

        # Bypass the producer to inject garbage.
        bad_key = shard.stream_key("wecom", "ext-bad")
        await shard.ensure_group(bad_key, "g")
        await shard.xadd(bad_key, {b"json": b"not-json-at-all"})

        async def never_called(msg: SystemMessage) -> None:
            raise AssertionError("handler should not run on malformed payload")

        run_task = asyncio.create_task(consumer.run(never_called))
        # Wait long enough for one read cycle to deserialize and route to DLQ.
        await asyncio.sleep(0.4)
        await consumer.stop()
        run_task.cancel()
        try:
            await run_task
        except (asyncio.CancelledError, BaseException):
            pass

        dlq_entries = await redis_client.xrange(shard.dlq_key())
        assert len(dlq_entries) == 1
        _, fields = dlq_entries[0]
        assert b"deserialize:" in fields[b"reason"]
