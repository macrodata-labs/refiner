from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from refiner.runtime.sinks import JsonlSink, ParquetSink
from refiner.pipeline import from_items
from refiner.sources.row import DictRow


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_jsonl_sink_writes_one_file_per_shard(tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path))
    counts1 = sink.write_block(
        [
            DictRow({"x": 1}).with_shard_id("a"),
            DictRow({"x": 2}).with_shard_id("b"),
        ]
    )
    counts2 = sink.write_block(pa.table({"x": [3, 4], "__shard_id": ["a", "b"]}))
    sink.on_shard_complete("a")
    sink.close()

    assert counts1 == {"a": 1, "b": 1}
    assert counts2 == {"a": 1, "b": 1}
    assert _read_jsonl(tmp_path / "a.jsonl") == [{"x": 1}, {"x": 3}]
    assert _read_jsonl(tmp_path / "b.jsonl") == [{"x": 2}, {"x": 4}]


def test_parquet_sink_writes_one_file_per_shard(tmp_path: Path) -> None:
    sink = ParquetSink(str(tmp_path))
    counts1 = sink.write_block(
        [
            DictRow({"x": 10}).with_shard_id("a"),
            DictRow({"x": 20}).with_shard_id("b"),
        ]
    )
    counts2 = sink.write_block(pa.table({"x": [30, 40], "__shard_id": ["a", "b"]}))
    sink.on_shard_complete("a")
    sink.close()

    assert counts1 == {"a": 1, "b": 1}
    assert counts2 == {"a": 1, "b": 1}

    table_a = pq.read_table(tmp_path / "a.parquet")
    table_b = pq.read_table(tmp_path / "b.parquet")
    assert table_a.schema.names == ["x"]
    assert table_b.schema.names == ["x"]
    assert table_a.to_pylist() == [{"x": 10}, {"x": 30}]
    assert table_b.to_pylist() == [{"x": 20}, {"x": 40}]


def test_pipeline_write_methods_attach_sink() -> None:
    pipeline = from_items([1, 2]).write_jsonl("/tmp/out")
    assert isinstance(pipeline.sink, JsonlSink)
    pipeline = from_items([1, 2]).write_parquet("/tmp/out")
    assert isinstance(pipeline.sink, ParquetSink)
