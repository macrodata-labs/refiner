from __future__ import annotations

from collections.abc import Iterator

from refiner.pipeline import from_items
from refiner.pipeline import from_source
from refiner.pipeline.data.shard import FilePart, Shard
from refiner.pipeline.data.shard import RowRangeDescriptor
from refiner.pipeline.data.row import DictRow, Row
from refiner.pipeline.sources.base import BaseSource


def test_from_items_yields_rows_across_shards() -> None:
    pipeline = from_items(
        [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}, {"x": 5}],
        items_per_shard=2,
    ).map(lambda row: {"y": int(row["x"]) * 10})

    out = list(pipeline.iter_rows())
    assert [int(r["y"]) for r in out] == [10, 20, 30, 40, 50]
    shards = pipeline.source.list_shards()
    assert len(shards) == 3
    assert isinstance(shards[0].descriptor, RowRangeDescriptor)
    assert shards[0].descriptor.start == 0
    assert shards[0].descriptor.end == 2
    assert isinstance(shards[2].descriptor, RowRangeDescriptor)
    assert shards[2].descriptor.start == 4
    assert shards[2].descriptor.end == 5


def test_from_items_wraps_primitives_in_items_column() -> None:
    pipeline = from_items([1, "x", True]).map(lambda row: {"v": row["item"]})
    out = list(pipeline.iter_rows())
    assert [r["v"] for r in out] == [1, "x", True]


class _CustomSource(BaseSource):
    def list_shards(self) -> list[Shard]:
        return [Shard.from_file_parts([FilePart(path="custom", start=0, end=2)])]

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        del shard
        yield DictRow({"x": 1})
        yield DictRow({"x": 2})


def test_from_source_accepts_custom_source() -> None:
    pipeline = from_source(_CustomSource()).map(lambda row: {"y": int(row["x"]) + 1})
    out = list(pipeline.iter_rows())
    assert [int(row["y"]) for row in out] == [2, 3]
