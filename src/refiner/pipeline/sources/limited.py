from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from refiner.pipeline.data.row import Row
from refiner.pipeline.data.shard import Shard, ShardGroupDescriptor
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.base import BaseSource, SourceUnit


class LimitedSource(BaseSource):
    """Apply one deterministic global row cap to another source.

    Limited runs expose one scheduling shard. That lets one worker consume the
    wrapped source shards in their deterministic order and stop at exactly the
    requested number of emitted source rows. Applying the cap independently in
    multiple workers would allow every worker to emit up to ``max_rows``.
    """

    claim_shards_sequentially = True

    def __init__(self, source: BaseSource, *, max_rows: int) -> None:
        if max_rows < 0:
            raise ValueError("max_rows must be >= 0")
        self.source = source
        self.max_rows = int(max_rows)
        self.name = source.name
        self._source_shards: tuple[Shard, ...] | None = None

    @property
    def schema(self) -> pa.Schema | None:
        return self.source.schema

    def required_refiner_extras(self) -> tuple[str, ...]:
        return self.source.required_refiner_extras()

    def describe(self) -> dict[str, Any]:
        description = dict(self.source.describe())
        description["max_rows"] = self.max_rows
        return description

    def _planned_source_shards(self) -> tuple[Shard, ...]:
        if self._source_shards is None:
            self._source_shards = tuple(self.source.list_shards())
        return self._source_shards

    def list_shards(self) -> list[Shard]:
        if self.max_rows == 0:
            return []
        source_shards = self._planned_source_shards()
        if not source_shards:
            return []
        return [
            Shard(
                descriptor=ShardGroupDescriptor(source_shards),
                global_ordinal=0,
            )
        ]

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        if not isinstance(descriptor, ShardGroupDescriptor):
            raise ValueError("LimitedSource requires a shard-group descriptor")

        remaining = self.max_rows
        if remaining == 0:
            return
        for source_shard in descriptor.shards:
            source_units = iter(self.source.read_shard(source_shard))
            try:
                for unit in source_units:
                    if isinstance(unit, Row):
                        yield unit
                        remaining -= 1
                        if remaining == 0:
                            return
                        continue
                    if not isinstance(unit, Tabular):
                        raise TypeError(f"Unsupported source unit type: {type(unit)!r}")
                    if unit.num_rows <= remaining:
                        yield unit
                        remaining -= unit.num_rows
                        if remaining == 0:
                            return
                        continue
                    row_indices = range(remaining) if unit.needs_row_indices else None
                    yield unit.with_table(
                        unit.table.slice(0, remaining),
                        row_indices=row_indices,
                    )
                    return
            finally:
                close = getattr(source_units, "close", None)
                if close is not None:
                    close()


def limit_source(source: BaseSource, max_rows: int | None) -> BaseSource:
    """Return ``source`` unchanged or wrapped with one global row cap."""
    if max_rows is None:
        return source
    return LimitedSource(source, max_rows=max_rows)


__all__ = ["LimitedSource", "limit_source"]
