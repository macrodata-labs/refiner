from __future__ import annotations

from collections.abc import Iterator

from refiner.ledger.shard import Shard
from refiner.pipeline import RefinerPipeline
from refiner.readers.base import BaseReader
from refiner.readers.row import DictRow, Row


class _LocalFakeReader(BaseReader):
    def __init__(self, shards: list[Shard], rows_by_shard_id: dict[str, list[Row]]):
        self._shards = shards
        self._rows_by_shard_id = rows_by_shard_id

    def list_shards(self) -> list[Shard]:
        return list(self._shards)

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        yield from self._rows_by_shard_id.get(shard.id, [])


def test_iter_rows_is_lazy_and_crosses_shards() -> None:
    s1 = Shard(path="a", start=0, end=1)
    s2 = Shard(path="b", start=0, end=1)
    rows = {
        s1.id: [DictRow({"x": 1}), DictRow({"x": 2})],
        s2.id: [DictRow({"x": 3}), DictRow({"x": 4})],
    }
    seen_batches: list[list[int]] = []

    def batch_fn(batch: list[Row]):
        seen_batches.append([int(r["x"]) for r in batch])
        return batch

    pipeline = (
        RefinerPipeline(source=_LocalFakeReader([s1, s2], rows))
        .map(lambda r: {"x": r["x"]})
        .map(batch_fn, batch_size=3)
    )

    it = iter(pipeline)
    first = next(it)
    assert first["x"] == 1
    # The first batch should include rows from both shards due to local global stream.
    assert seen_batches[0] == [1, 2, 3]


def test_materialize_and_take() -> None:
    s1 = Shard(path="a", start=0, end=1)
    s2 = Shard(path="b", start=0, end=1)
    rows = {
        s1.id: [DictRow({"x": 1}), DictRow({"x": 2})],
        s2.id: [DictRow({"x": 3})],
    }
    pipeline = RefinerPipeline(source=_LocalFakeReader([s1, s2], rows)).map(
        lambda r: {"y": r["x"] * 10}
    )

    assert [r["y"] for r in pipeline.take(2)] == [10, 20]
    assert [r["y"] for r in pipeline.materialize()] == [10, 20, 30]


def test_batch_groups_split_on_increasing_batch_size() -> None:
    s = Shard(path="a", start=0, end=1)
    rows = {s.id: [DictRow({"x": i}) for i in range(10)]}
    seen_4: list[int] = []
    seen_8: list[int] = []

    def b4(batch: list[Row]):
        seen_4.append(len(batch))
        return batch

    def b8(batch: list[Row]):
        seen_8.append(len(batch))
        return batch

    pipeline = (
        RefinerPipeline(source=_LocalFakeReader([s], rows))
        .map(b4, batch_size=4)
        .map(b8, batch_size=8)
    )

    out = list(pipeline.iter_rows())
    assert len(out) == 10
    assert seen_4 == [4, 4, 2]
    assert seen_8 == [8, 2]


def test_downstream_batch_waits_after_upstream_drop() -> None:
    s = Shard(path="a", start=0, end=1)
    rows = {s.id: [DictRow({"x": i}) for i in range(8)]}
    seen_b2: list[int] = []

    def drop_to_one(batch: list[Row]):
        if not batch:
            return []
        return [batch[0]]

    def b2(batch: list[Row]):
        seen_b2.append(len(batch))
        return batch

    pipeline = (
        RefinerPipeline(source=_LocalFakeReader([s], rows))
        .map(drop_to_one, batch_size=4)
        .map(b2, batch_size=2)
    )

    out = list(pipeline.iter_rows())
    assert len(out) == 2
    # b2 should run once with a full batch collected across two upstream outputs.
    assert seen_b2 == [2]
