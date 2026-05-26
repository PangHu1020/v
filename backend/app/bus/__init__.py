"""Bus layer: Redis Streams reactive-inbound message bus.

Sharded by ``(channel, channel_user_id)`` so each conversation is processed
strictly serially while distinct conversations parallelize across consumers.
"""

from backend.app.bus.consumer import BusConsumer
from backend.app.bus.messages import SystemMessage
from backend.app.bus.producer import BusProducer
from backend.app.bus.shard import RedisStreamShard

__all__ = ["BusConsumer", "BusProducer", "RedisStreamShard", "SystemMessage"]
