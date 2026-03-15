from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from refiner.pipeline.data.shard import RowRangeDescriptor, Shard
from refiner.pipeline.sources.base import BaseSource
from refiner.pipeline.data.row import DictRow, Row

_DEFAULT_ITEMS_PER_SHARD = 1_000


class ItemsSource(BaseSource):
    """In-memory source built from Python row mappings."""

    name = "from_items"

    def __init__(
        self,
        items: Sequence[Any],
        *,
        items_per_shard: int = _DEFAULT_ITEMS_PER_SHARD,
    ) -> None:
        if items_per_shard <= 0:
            raise ValueError("items_per_shard must be > 0")

        self._rows = tuple(_normalize_item(item) for item in items)
        self._row_count = len(self._rows)
        self._items_per_shard = items_per_shard

    def list_shards(self) -> list[Shard]:
        shards: list[Shard] = []
        for start in range(0, self._row_count, self._items_per_shard):
            end = min(self._row_count, start + self._items_per_shard)
            shards.append(
                Shard(
                    descriptor=RowRangeDescriptor(
                        start=start,
                        end=end,
                    ),
                    global_ordinal=len(shards),
                )
            )
        return shards

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        descriptor = shard.descriptor
        if not isinstance(descriptor, RowRangeDescriptor):
            raise TypeError("ItemsSource requires row-range shards")
        start = int(descriptor.start)
        end = int(descriptor.end)
        if start < 0 or end < start or end > self._row_count:
            raise ValueError(
                f"Invalid items shard range [{start}, {end}) for {self._row_count} rows"
            )
        for row in self._rows[start:end]:
            yield DictRow(data=row)

    def describe(self) -> dict[str, Any]:
        return {
            "rows": self._row_count,
            "items_per_shard": self._items_per_shard,
        }


def _normalize_item(item: Any) -> dict[str, Any]:
    if isinstance(item, Row):
        return dict(item)
    if isinstance(item, Mapping):
        return dict(item)
    return {"item": item}


__all__ = [
    "ItemsSource",
]
