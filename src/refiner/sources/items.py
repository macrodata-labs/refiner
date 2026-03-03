from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from refiner.ledger.shard import Shard
from refiner.sources.base import BaseSource
from refiner.sources.row import DictRow, Row

_DEFAULT_ITEMS_SHARD_SIZE_ROWS = 1_000
_ITEMS_SOURCE_PATH = "memory://items"


class ItemsSource(BaseSource):
    """In-memory source built from Python row mappings."""

    name = "from_items"

    def __init__(
        self,
        items: Sequence[Any],
        *,
        shard_size_rows: int = _DEFAULT_ITEMS_SHARD_SIZE_ROWS,
    ) -> None:
        if shard_size_rows <= 0:
            raise ValueError("shard_size_rows must be > 0")

        self._rows = tuple(_normalize_item(item) for item in items)
        self._row_count = len(self._rows)
        self._shard_size_rows = shard_size_rows
        self._source_path = _ITEMS_SOURCE_PATH

    def list_shards(self) -> list[Shard]:
        shards: list[Shard] = []
        for start in range(0, self._row_count, self._shard_size_rows):
            end = min(self._row_count, start + self._shard_size_rows)
            shards.append(Shard(path=self._source_path, start=start, end=end))
        return shards

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        if shard.path != self._source_path:
            raise ValueError(f"Unknown items shard path: {shard.path!r}")
        start = int(shard.start)
        end = int(shard.end)
        if start < 0 or end < start or end > self._row_count:
            raise ValueError(
                f"Invalid items shard range [{start}, {end}) for {self._row_count} rows"
            )
        for row in self._rows[start:end]:
            yield DictRow(data=row)

    def describe(self) -> dict[str, Any]:
        return {
            "rows": self._row_count,
            "shard_size_rows": self._shard_size_rows,
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
