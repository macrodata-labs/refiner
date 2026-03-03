from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pytest

from refiner.pipeline import from_items
from refiner import col


def test_vectorized_pipeline_ops_execute_in_order() -> None:
    out = (
        from_items(
            [
                {"x": 1, "text": " A "},
                {"x": 2, "text": "Bee"},
                {"x": 3, "text": "cee"},
            ]
        )
        .filter(col("x") >= 2)
        .with_columns(x2=col("x") * 2, text_clean=col("text").str.strip().str.lower())
        .drop("text")
        .rename(x2="score")
        .cast(score="int64")
        .select("score", "text_clean")
        .materialize()
    )

    assert [int(r["score"]) for r in out] == [4, 6]
    assert [str(r["text_clean"]) for r in out] == ["bee", "cee"]


def test_vectorized_and_row_udf_segments_interoperate() -> None:
    out = (
        from_items([{"x": 1}, {"x": 2}, {"x": 3}])
        .with_columns(y=col("x") + 1)
        .map(lambda row: {"z": int(row["y"]) * 10})
        .with_column("w", col("z") + 5)
        .select("x", "w")
        .materialize()
    )

    assert [int(r["x"]) for r in out] == [1, 2, 3]
    assert [int(r["w"]) for r in out] == [25, 35, 45]


def test_datetime_namespace_to_date_and_year() -> None:
    ts = datetime(2025, 1, 2, 15, 30, tzinfo=timezone.utc)
    out = (
        from_items([{"ts": ts}])
        .with_columns(
            d=col("ts").datetime.to_date(),
            y=col("ts").datetime.year(),
        )
        .select("d", "y")
        .materialize()
    )

    assert str(out[0]["d"]) == "2025-01-02"
    assert int(out[0]["y"]) == 2025


def test_cast_unknown_column_raises() -> None:
    with pytest.raises(KeyError):
        from_items([{"x": 1}]).cast(y="int64").materialize()


def test_execute_blocks_keeps_arrow_for_vectorized_segment() -> None:
    pipeline = from_items([{"x": 1}, {"x": 2}]).with_column("y", col("x") + 1)
    blocks = list(pipeline.execute(pipeline.source.read()))
    assert blocks
    assert all(isinstance(block, (pa.RecordBatch, pa.Table)) for block in blocks)
    assert sum(int(block.num_rows) for block in blocks) == 2


def test_execute_blocks_switches_back_to_arrow_after_row_segment() -> None:
    pipeline = (
        from_items([{"x": 1}, {"x": 2}])
        .with_column("y", col("x") + 1)
        .map(lambda row: {"z": int(row["y"]) * 10})
        .with_column("w", col("z") + 5)
    )
    blocks = list(pipeline.execute(pipeline.source.read()))
    assert blocks
    assert all(isinstance(block, (pa.RecordBatch, pa.Table)) for block in blocks)
    assert sum(int(block.num_rows) for block in blocks) == 2
