"""Unit tests for ``backend.app.bus.shard``."""

from __future__ import annotations

import fakeredis.aioredis
import pytest

from backend.app.bus.shard import RedisStreamShard


@pytest.fixture
async def redis_client() -> fakeredis.aioredis.FakeRedis:
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


@pytest.fixture
def shard(redis_client: fakeredis.aioredis.FakeRedis) -> RedisStreamShard:
    return RedisStreamShard(redis_client, prefix="bus", shard_count=64)


class TestShardIndex:
    def test_deterministic_across_calls(self, shard: RedisStreamShard) -> None:
        first = shard.shard_index("wecom", "ext-1")
        second = shard.shard_index("wecom", "ext-1")
        assert first == second

    def test_different_users_can_land_on_different_shards(self, shard: RedisStreamShard) -> None:
        # 200 distinct users will, with overwhelming probability, hit at least
        # 2 distinct shards out of 64. We just need to assert non-degenerate
        # distribution.
        shards = {shard.shard_index("wecom", f"ext-{i}") for i in range(200)}
        assert len(shards) > 1

    def test_index_in_range(self, shard: RedisStreamShard) -> None:
        for i in range(50):
            idx = shard.shard_index("feishu", f"u-{i}")
            assert 0 <= idx < 64

    def test_channel_is_part_of_hash(self, shard: RedisStreamShard) -> None:
        # The same channel_user_id under different channels should hash
        # independently. We don't require they always differ — just that the
        # function inputs include the channel.
        wecom_idx = [shard.shard_index("wecom", f"x-{i}") for i in range(50)]
        feishu_idx = [shard.shard_index("feishu", f"x-{i}") for i in range(50)]
        assert wecom_idx != feishu_idx


class TestStreamKey:
    def test_format(self, shard: RedisStreamShard) -> None:
        key = shard.stream_key("wecom", "ext-1")
        assert key.startswith("bus:")
        idx = int(key.split(":")[1])
        assert 0 <= idx < 64

    def test_all_stream_keys(self, shard: RedisStreamShard) -> None:
        keys = shard.all_stream_keys()
        assert len(keys) == 64
        assert keys[0] == "bus:0"
        assert keys[-1] == "bus:63"

    def test_dlq_key(self, shard: RedisStreamShard) -> None:
        assert shard.dlq_key() == "bus:dlq"


class TestShardCountValidation:
    def test_zero_rejected(self, redis_client: fakeredis.aioredis.FakeRedis) -> None:
        with pytest.raises(ValueError):
            RedisStreamShard(redis_client, prefix="bus", shard_count=0)


class TestRedisOperations:
    async def test_xadd_then_read_via_group(
        self,
        shard: RedisStreamShard,
    ) -> None:
        key = shard.stream_key("wecom", "ext-1")
        await shard.ensure_group(key, "g")
        entry_id = await shard.xadd(key, {b"json": b'{"v": 1}'})
        assert entry_id

        entries = await shard.xreadgroup("g", "c1", [key], count=1, block_ms=10)
        assert len(entries) == 1
        _stream_key, batch = entries[0]
        msg_id, fields = batch[0]
        assert fields[b"json"] == b'{"v": 1}'

        acked = await shard.xack(key, "g", msg_id)
        assert acked == 1

    async def test_ensure_group_is_idempotent(
        self,
        shard: RedisStreamShard,
    ) -> None:
        key = shard.stream_key("wecom", "ext-1")
        await shard.ensure_group(key, "g")
        await shard.ensure_group(key, "g")  # second call must not raise
