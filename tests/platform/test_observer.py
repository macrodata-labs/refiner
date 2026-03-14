from __future__ import annotations

from refiner.pipeline.data.shard import FilePart, FilePartsDescriptor, Shard
from refiner.platform.client import compile_shard_descriptors
from refiner.platform.client import ShardDescriptor


def test_compile_shard_descriptors_uses_stable_shard_id() -> None:
    shard = Shard(
        descriptor=FilePartsDescriptor(
            (FilePart(path="s3://bucket/file.parquet", start=10, end=20),)
        )
    )
    descriptors = compile_shard_descriptors([shard])
    assert descriptors == [
        ShardDescriptor(
            shard_id=shard.id,
            global_ordinal=None,
            start_key=shard.start_key,
            end_key=shard.end_key,
            descriptor=shard.descriptor.to_dict(),
        )
    ]
