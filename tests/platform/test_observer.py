from __future__ import annotations

from refiner.pipeline.data.shard import Shard
from refiner.platform.client import compile_shard_descriptors
from refiner.platform.client import ShardDescriptor


def test_compile_shard_descriptors_uses_stable_shard_id() -> None:
    shard = Shard(path="s3://bucket/file.parquet", start=10, end=20)
    descriptors = compile_shard_descriptors([shard])
    assert descriptors == [
        ShardDescriptor(
            shard_id=shard.id,
            path="s3://bucket/file.parquet",
            start=10,
            end=20,
        )
    ]
