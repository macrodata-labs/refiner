from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pytest

from refiner.pipeline.data.block import TabularBlock
import refiner.pipeline.pipeline as pipeline_module
import refiner.execution.engine as engine_module
from refiner.pipeline import from_items
from refiner import col, if_else


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
    tabular_blocks = [block for block in blocks if isinstance(block, TabularBlock)]
    assert len(tabular_blocks) == len(blocks)
    assert sum(int(block.table.num_rows) for block in tabular_blocks) == 2


def test_execute_blocks_switches_back_to_arrow_after_row_segment() -> None:
    pipeline = (
        from_items([{"x": 1}, {"x": 2}])
        .with_column("y", col("x") + 1)
        .map(lambda row: {"z": int(row["y"]) * 10})
        .with_column("w", col("z") + 5)
    )
    blocks = list(pipeline.execute(pipeline.source.read()))
    assert blocks
    tabular_blocks = [block for block in blocks if isinstance(block, TabularBlock)]
    assert len(tabular_blocks) == len(blocks)
    assert sum(int(block.table.num_rows) for block in tabular_blocks) == 2


def test_execute_caches_compiled_segments(monkeypatch) -> None:
    compile_calls = 0
    original_compile = pipeline_module.compile_segments

    def _counting_compile(steps):
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(steps)

    monkeypatch.setattr(pipeline_module, "compile_segments", _counting_compile)

    pipeline = from_items([{"x": 1}, {"x": 2}]).with_column("y", col("x") + 1)
    list(pipeline.execute(pipeline.source.read()))
    list(pipeline.execute(pipeline.source.read()))

    assert compile_calls == 1


def test_max_vectorized_block_bytes_can_force_smaller_blocks() -> None:
    pipeline = (
        from_items([{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}])
        .with_column("y", col("x") + 1)
        .with_max_vectorized_block_bytes(1)
    )
    blocks = list(pipeline.execute(pipeline.source.read()))
    assert blocks
    tabular_blocks = [block for block in blocks if isinstance(block, TabularBlock)]
    assert len(tabular_blocks) == len(blocks)
    assert all(int(block.table.num_rows) <= 1 for block in tabular_blocks)


def test_vectorized_chunk_shrink_is_run_local(monkeypatch) -> None:
    calls: list[int] = []
    original_rows_to_block = engine_module.rows_to_block

    def _rows_to_block_with_oom(rows):
        calls.append(len(rows))
        if len(rows) > 2:
            raise pa.ArrowMemoryError("oom")
        return original_rows_to_block(rows)

    monkeypatch.setattr(engine_module, "rows_to_block", _rows_to_block_with_oom)

    pipeline = (
        from_items([{"x": i} for i in range(8)])
        .with_column("y", col("x") + 1)
        .with_max_vectorized_block_bytes(1_000_000)
    )

    first_run_start = len(calls)
    out1 = pipeline.materialize()
    second_run_start = len(calls)
    out2 = pipeline.materialize()

    assert len(out1) == 8
    assert len(out2) == 8
    assert calls[first_run_start] == 8
    assert calls[second_run_start] == 8


def test_with_max_vectorized_block_bytes_validates_positive() -> None:
    with pytest.raises(ValueError):
        from_items([{"x": 1}]).with_max_vectorized_block_bytes(0)


def test_vectorized_expression_extensions() -> None:
    out = (
        from_items(
            [
                {"x": 1, "s": "foo1", "y": None, "z": 1.2},
                {"x": 2, "s": "bar2", "y": 5, "z": 2.6},
                {"x": 3, "s": "baz3", "y": None, "z": -3.1},
            ]
        )
        .with_columns(
            in_set=col("x").is_in([1, 3]),
            between_2_3=col("x").between(2, 3),
            starts=col("s").str.startswith("ba"),
            ends=col("s").str.endswith("3"),
            has_digit=col("s").str.regex_contains(r"\d"),
            no_digit=col("s").str.regex_replace(r"\d", ""),
            y_filled=col("y").fill_null(0),
            x_null_if_2=col("x").null_if(2),
            bucket=if_else(col("x") > 1, "gt1", "le1"),
            z_abs=col("z").abs(),
            z_floor=col("z").floor(),
            z_ceil=col("z").ceil(),
            z_round=col("z").round(0),
            z_clip=col("z").clip(min_value=1.5, max_value=2.5),
        )
        .select(
            "in_set",
            "between_2_3",
            "starts",
            "ends",
            "has_digit",
            "no_digit",
            "y_filled",
            "x_null_if_2",
            "bucket",
            "z_abs",
            "z_floor",
            "z_ceil",
            "z_round",
            "z_clip",
        )
        .materialize()
    )

    assert [bool(r["in_set"]) for r in out] == [True, False, True]
    assert [bool(r["between_2_3"]) for r in out] == [False, True, True]
    assert [bool(r["starts"]) for r in out] == [False, True, True]
    assert [bool(r["ends"]) for r in out] == [False, False, True]
    assert [bool(r["has_digit"]) for r in out] == [True, True, True]
    assert [str(r["no_digit"]) for r in out] == ["foo", "bar", "baz"]
    assert [int(r["y_filled"]) for r in out] == [0, 5, 0]
    assert [r["x_null_if_2"] for r in out] == [1, None, 3]
    assert [str(r["bucket"]) for r in out] == ["le1", "gt1", "gt1"]
    assert [float(r["z_abs"]) for r in out] == pytest.approx([1.2, 2.6, 3.1])
    assert [int(r["z_floor"]) for r in out] == [1, 2, -4]
    assert [int(r["z_ceil"]) for r in out] == [2, 3, -3]
    assert [float(r["z_round"]) for r in out] == pytest.approx([1.0, 3.0, -3.0])
    assert [float(r["z_clip"]) for r in out] == pytest.approx([1.5, 2.5, 1.5])
