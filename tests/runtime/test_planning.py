from __future__ import annotations

from collections.abc import Iterator

from refiner.ledger.shard import Shard
from refiner.pipeline import RefinerPipeline, from_items
from refiner.sources.readers.base import BaseReader
from refiner.sources.row import DictRow, Row
from refiner.runtime.planning import compile_pipeline_plan


class _FakeReader(BaseReader):
    def __init__(self) -> None:
        super().__init__(inputs=[])

    @property
    def files(self) -> list[str]:  # type: ignore[override]
        return ["data/a.parquet"]

    def list_shards(self) -> list[Shard]:
        return [Shard(path="data/a.parquet", start=0, end=10)]

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        del shard
        yield DictRow({"x": 1})


def test_compile_pipeline_plan_includes_reader_and_steps() -> None:
    pipeline = (
        RefinerPipeline(_FakeReader())
        .map(lambda row: {"x": row["x"]})
        .batch_map(lambda rows: rows, batch_size=2)
        .flat_map(lambda row: [row])
    )

    payload = compile_pipeline_plan(pipeline)
    stages = payload["stages"]
    assert len(stages) == 1
    steps = stages[0]["steps"]
    assert steps[0]["type"] == "source"
    assert [step["name"] for step in steps[1:]] == ["map", "batch_map", "flat_map"]
    assert steps[2]["args"]["batch_size"] == 2
    assert "code" not in steps[0]


def test_compile_pipeline_plan_makes_step_names_unique() -> None:
    pipeline = (
        RefinerPipeline(_FakeReader())
        .filter(lambda row: True)
        .flat_map(lambda row: [row])
    )

    payload = compile_pipeline_plan(pipeline)
    steps = payload["stages"][0]["steps"]
    assert steps[0]["name"].startswith("read_")
    assert [step["name"] for step in steps[1:]] == ["filter", "flat_map"]


def test_compile_pipeline_plan_dedupes_same_top_level_op_names() -> None:
    pipeline = (
        RefinerPipeline(_FakeReader()).filter(lambda row: True).filter(lambda row: True)
    )

    payload = compile_pipeline_plan(pipeline)
    steps = payload["stages"][0]["steps"]
    assert [step["name"] for step in steps[1:]] == ["filter", "filter_2"]


def test_compile_pipeline_plan_includes_from_items_metadata() -> None:
    pipeline = from_items([{"x": 1}, {"x": 2}, {"x": 3}], shard_size_rows=2)
    payload = compile_pipeline_plan(pipeline)
    source_step = payload["stages"][0]["steps"][0]
    assert source_step["name"] == "from_items"
    assert source_step["args"]["rows"] == 3
    assert source_step["args"]["shard_size_rows"] == 2
