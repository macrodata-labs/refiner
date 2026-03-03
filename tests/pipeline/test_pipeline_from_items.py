from __future__ import annotations

from refiner.pipeline import from_items


def test_from_items_yields_rows_across_shards() -> None:
    pipeline = from_items(
        [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}, {"x": 5}],
        shard_size_rows=2,
    ).map(lambda row: {"y": int(row["x"]) * 10})

    out = list(pipeline.iter_rows())
    assert [int(r["y"]) for r in out] == [10, 20, 30, 40, 50]
    shards = pipeline.source.list_shards()
    assert len(shards) == 3
    assert shards[0].start == 0
    assert shards[0].end == 2
    assert shards[2].start == 4
    assert shards[2].end == 5


def test_from_items_wraps_primitives_in_items_column() -> None:
    pipeline = from_items([1, "x", True]).map(lambda row: {"v": row["item"]})
    out = list(pipeline.iter_rows())
    assert [r["v"] for r in out] == [1, "x", True]
