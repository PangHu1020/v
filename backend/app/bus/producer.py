"""Bus producer: serializes a SystemMessage and XADDs it to its shard."""

from __future__ import annotations

import orjson

from backend.app.bus.messages import SystemMessage
from backend.app.bus.shard import RedisStreamShard
from backend.v.utils.logging import get_logger

_log = get_logger("bus.producer")


class BusProducer:
    """Push normalized messages onto the sharded Redis Streams bus."""

    def __init__(self, shard: RedisStreamShard) -> None:
        self._shard = shard

    async def enqueue(self, message: SystemMessage) -> str:
        """Serialize the message and append it to the appropriate shard stream.

        Args:
            message: A normalized inbound message from a channel adapter.

        Returns:
            The Redis stream entry id assigned to the entry, decoded as ASCII.
        """
        key = self._shard.stream_key(message.channel, message.channel_user_id)
        # ``model_dump_json`` preserves the Pydantic schema (including datetimes
        # and the discriminated channel literal) and round-trips cleanly.
        payload = message.model_dump_json().encode("utf-8")
        entry_id_bytes = await self._shard.xadd(key, {b"json": payload})
        entry_id = entry_id_bytes.decode("ascii")
        _log.info(
            "bus.enqueue",
            stream=key,
            entry_id=entry_id,
            channel=message.channel,
            channel_user_id=message.channel_user_id,
            dedup_key=message.dedup_key,
        )
        return entry_id


def serialize_message(message: SystemMessage) -> bytes:
    """Standalone serializer kept for callers that bypass the producer (tests)."""
    return orjson.dumps(message.model_dump(mode="json"))
