from __future__ import annotations

import json

import pyarrow.parquet as pq

from refiner import col
from refiner.pipeline.data.row import DictRow
from refiner.pipeline import from_items
from refiner.pipeline.sinks import JsonlSink
from refiner.worker.context import RunHandle


def test_iter_rows_ignores_sink(tmp_path) -> None:
    pipeline = from_items([{"x": 1}, {"x": 2}], items_per_shard=1).write_jsonl(tmp_path)
    out = list(pipeline.iter_rows())
    assert [int(row["x"]) for row in out] == [1, 2]
    assert list(tmp_path.iterdir()) == []


def test_launch_local_writes_jsonl_per_shard(tmp_path) -> None:
    output_dir = tmp_path / "jsonl-output"
    pipeline = (
        from_items([{"x": 1}, {"x": 2}, {"x": 3}], items_per_shard=2)
        .map(lambda row: {"x": int(row["x"]) * 10})
        .write_jsonl(output_dir)
    )

    stats = pipeline.launch_local(
        name="jsonl-sink", num_workers=1, rundir=str(tmp_path / "run")
    )

    assert stats.completed == 2
    written = sorted(path.name for path in output_dir.iterdir())
    assert len(written) == 2
    assert all("__w" in name for name in written)
    assert all(name.endswith(".jsonl") for name in written)


def test_launch_local_writes_parquet_per_shard(tmp_path) -> None:
    output_dir = tmp_path / "parquet-output"
    pipeline = (
        from_items([{"x": 1}, {"x": 2}, {"x": 3}], items_per_shard=2)
        .map(lambda row: {"x": int(row["x"]) * 10})
        .write_parquet(output_dir)
    )

    stats = pipeline.launch_local(
        name="parquet-sink", num_workers=1, rundir=str(tmp_path / "run")
    )

    assert stats.completed == 2
    written = sorted(path for path in output_dir.iterdir() if path.suffix == ".parquet")
    assert len(written) == 2
    assert all("__w" in path.name for path in written)
    values = []
    for path in written:
        table = pq.read_table(path)
        values.extend(int(value) for value in table.column("x").to_pylist())
    assert sorted(values) == [10, 20, 30]


def test_launch_local_vectorized_filter_with_sink_completes_shards(tmp_path) -> None:
    output_dir = tmp_path / "vectorized-output"
    pipeline = (
        from_items([{"x": 1}, {"x": 2}, {"x": 3}], items_per_shard=2)
        .filter(col("x") > 1)
        .write_jsonl(output_dir)
    )

    stats = pipeline.launch_local(
        name="vectorized-jsonl-sink",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    assert stats.completed == 2
    written = sorted(path for path in output_dir.iterdir() if path.suffix == ".jsonl")
    assert len(written) == 2
    assert all("__w" in path.name for path in written)


def test_jsonl_sink_uses_local_worker_suffix_outside_runtime(tmp_path) -> None:
    sink = JsonlSink(tmp_path)
    sink.write_block([DictRow({"x": 1}, shard_id="abc")])
    sink.on_shard_complete("abc")

    written = sorted(tmp_path.iterdir())
    assert [path.name for path in written] == [
        f"abc__w{RunHandle.worker_token_for('local')}.jsonl"
    ]
    assert json.loads(written[0].read_text(encoding="utf-8")) == {"x": 1}
