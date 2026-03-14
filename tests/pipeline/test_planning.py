from __future__ import annotations

from collections.abc import Iterator

from refiner.pipeline.data.shard import Shard
from refiner import col
from refiner.pipeline import RefinerPipeline, from_items
from refiner.pipeline.sources.readers.base import BaseReader
from refiner.pipeline.data.row import DictRow, Row
from refiner.pipeline.planning import (
    _extract_lambda_source,
    compile_pipeline_plan,
    plan_pipeline_stages,
)


class _FakeReader(BaseReader):
    def __init__(self) -> None:
        super().__init__(inputs=[])

    @property
    def files(self) -> list[str]:
        return ["data/a.parquet"]

    def list_shards(self) -> list[Shard]:
        return [Shard(path="data/a.parquet", start=0, end=10)]

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        del shard
        yield DictRow({"x": 1})


def _score_filter_lambda():
    return lambda row: int(row["score"]) >= 15


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
    assert "fn" not in steps[0].get("args", {})
    assert "lambda row" in steps[1]["args"]["fn"]
    assert steps[1]["args"]["__meta"]["fn"] == "code"
    assert "lambda rows" in steps[2]["args"]["fn"]
    assert steps[2]["args"]["__meta"]["fn"] == "code"


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
    assert steps[1]["type"] == "filter"
    assert "lambda row: True" in steps[1]["args"]["fn"]


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


def test_compile_pipeline_plan_flattens_vectorized_segment_ops() -> None:
    payload = (
        from_items([{"x": 1}, {"x": 2}])
        .filter(col("x") > 1)
        .with_columns(y=col("x") + 10)
        .select("y")
    )
    plan = compile_pipeline_plan(payload)
    steps = plan["stages"][0]["steps"]
    assert [step["name"] for step in steps] == [
        "from_items",
        "filter",
        "with_columns",
        "select",
    ]
    assert steps[1]["type"] == "filter_expr"
    assert steps[2]["type"] == "with_columns"
    assert steps[3]["type"] == "select"
    assert "expression" in steps[1]["args"]
    assert "callable" not in steps[1]
    assert steps[2]["args"] == {"y": "(col('x') + 10)"}
    assert "callable" not in steps[2]


def test_compile_pipeline_plan_uses_named_callable_for_step_name() -> None:
    def duplicate_selected(row):
        return {"x": row["x"], "dup": True}

    pipeline = RefinerPipeline(_FakeReader()).map(duplicate_selected)
    payload = compile_pipeline_plan(pipeline)
    steps = payload["stages"][0]["steps"]
    assert steps[1]["name"] == "duplicate_selected"
    assert steps[1]["type"] == "row_map"


def test_extract_lambda_source_handles_chained_call_fragment() -> None:
    fn = _score_filter_lambda()
    source = '.filter(lambda row: int(row["score"]) >= 15)'
    assert _extract_lambda_source(source, fn) == 'lambda row: int(row["score"]) >= 15'


def test_extract_lambda_source_matches_exact_lambda_when_multiple_present() -> None:
    fn = _score_filter_lambda()
    source = (
        'pipeline.map(lambda row: row["score"]).filter('
        'lambda row: int(row["score"]) >= 15)'
    )
    assert _extract_lambda_source(source, fn) == 'lambda row: int(row["score"]) >= 15'


def test_plan_pipeline_stages_returns_single_placeholder_stage() -> None:
    pipeline = from_items([{"x": 1}])
    stages = plan_pipeline_stages(pipeline, default_num_workers=3)

    assert len(stages) == 1
    assert stages[0].index == 0
    assert stages[0].name == "stage_0"
    assert stages[0].pipeline is pipeline
    assert stages[0].compute.num_workers == 3
