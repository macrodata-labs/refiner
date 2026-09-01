from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from refiner.pipeline.data.shard import Shard, ShardGroupDescriptor
from refiner.pipeline.sources.base import BaseSource, SourceUnit


class GlobalValidationSource(BaseSource):
    """Collapse a source plan into one retryable unit for exact global checks."""

    def __init__(self, source: BaseSource) -> None:
        self.source = source
        self.name = source.name
        self._source_shards: tuple[Shard, ...] | None = None

    @property
    def schema(self) -> pa.Schema | None:
        return self.source.schema

    def required_refiner_extras(self) -> tuple[str, ...]:
        return self.source.required_refiner_extras()

    def describe(self) -> dict[str, Any]:
        description = dict(self.source.describe())
        description["global_validation"] = True
        return description

    def _planned_source_shards(self) -> tuple[Shard, ...]:
        if self._source_shards is None:
            self._source_shards = tuple(self.source.list_shards())
        return self._source_shards

    def list_shards(self) -> list[Shard]:
        return [
            Shard(
                descriptor=ShardGroupDescriptor(self._planned_source_shards()),
                global_ordinal=0,
            )
        ]

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        if not isinstance(descriptor, ShardGroupDescriptor):
            raise ValueError("GlobalValidationSource requires a shard-group descriptor")
        for source_shard in descriptor.shards:
            source_units = iter(self.source.read_shard(source_shard))
            try:
                yield from source_units
            finally:
                close = getattr(source_units, "close", None)
                if close is not None:
                    close()


def global_validation_source(source: BaseSource) -> GlobalValidationSource:
    if isinstance(source, GlobalValidationSource):
        return source
    return GlobalValidationSource(source)


def unwrap_global_validation_source(source: BaseSource) -> BaseSource:
    if isinstance(source, GlobalValidationSource):
        return source.source
    return source


__all__ = [
    "GlobalValidationSource",
    "global_validation_source",
    "unwrap_global_validation_source",
]
