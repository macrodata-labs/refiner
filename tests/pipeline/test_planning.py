from __future__ import annotations

from collections.abc import Iterator

import refiner as rf
import pytest
from refiner.pipeline.data.shard import FilePart, Shard
from refiner import col
from refiner.pipeline import FollowupStage, RefinerPipeline, from_items
from refiner.pipeline.sources.readers.base import BaseReader
from refiner.pipeline.data.row import DictRow, Row
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.planning import (
    StageComputeRequirements,
    _extract_lambda_source,
    compile_pipeline_plan,
    plan_pipeline_stages,
)
from refiner.robotics import motion_trim


class FakeReader(BaseReader):
    def __init__(self) -> None:
        super().__init__(inputs=[])

    @property
    def files(self) -> list[str]:
        return ["data/a.parquet"]

    def list_shards(self) -> list[Shard]:
        return [
            Shard.from_file_parts([FilePart(path="data/a.parquet", start=0, end=10)])
        ]

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        del shard
        yield DictRow({"x": 1})


class UndescribedSink(BaseSink):
    def write_block(self, block):
        del block
        return {}, 0


class MultiStageSink(UndescribedSink):
    def followup_stages(self) -> tuple[FollowupStage, ...]:
        return (
            FollowupStage.from_sink(
                name="index",
                sink=UndescribedSink(),
                num_workers=2,
                cpus_per_worker=4,
            ),
            FollowupStage.from_sink(name="publish", sink=UndescribedSink()),
        )


def _score_filter_lambda():
    return lambda row: int(row["score"]) >= 15


def test_compile_pipeline_plan_includes_reader_and_steps() -> None:
    pipeline = (
        RefinerPipeline(FakeReader())
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
        RefinerPipeline(FakeReader())
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
        RefinerPipeline(FakeReader()).filter(lambda row: True).filter(lambda row: True)
    )

    payload = compile_pipeline_plan(pipeline)
    steps = payload["stages"][0]["steps"]
    assert [step["name"] for step in steps[1:]] == ["filter", "filter_2"]


def test_compile_pipeline_plan_includes_from_items_metadata() -> None:
    pipeline = from_items([{"x": 1}, {"x": 2}, {"x": 3}], items_per_shard=2)
    payload = compile_pipeline_plan(pipeline)
    source_step = payload["stages"][0]["steps"][0]
    assert source_step["name"] == "from_items"
    assert source_step["args"]["rows"] == 3
    assert source_step["args"]["items_per_shard"] == 2


def test_compile_pipeline_plan_flattens_vectorized_segment_ops() -> None:
    payload = (
        from_items([{"x": 1}, {"x": 2}])
        .filter(col("x") > 1)
        .with_columns(y=col("x") + 10)
        .select("y")
        .cast(y=rf.datatype.video_path())
    )
    plan = compile_pipeline_plan(payload)
    steps = plan["stages"][0]["steps"]
    assert [step["name"] for step in steps] == [
        "from_items",
        "filter",
        "with_columns",
        "select",
        "cast",
    ]
    assert steps[1]["type"] == "filter_expr"
    assert steps[2]["type"] == "with_columns"
    assert steps[3]["type"] == "select"
    assert steps[4]["type"] == "cast"
    assert "expression" in steps[1]["args"]
    assert "callable" not in steps[1]
    assert steps[2]["args"] == {"y": "(col('x') + 10)"}
    assert steps[4]["args"]["dtypes"] == {
        "y": {"type": "string", "metadata": {"asset_type": "video"}}
    }
    assert "callable" not in steps[2]


def test_compile_pipeline_plan_includes_map_table_step() -> None:
    payload = from_items([{"x": 1}, {"x": 2}]).map_table(lambda table: table)

    plan = compile_pipeline_plan(payload)
    steps = plan["stages"][0]["steps"]

    assert [step["name"] for step in steps] == ["from_items", "map_table"]
    assert steps[1]["type"] == "table_map"
    assert "fn" in steps[1]["args"]


def test_compile_pipeline_plan_uses_named_callable_for_step_name() -> None:
    def duplicate_selected(row):
        return {"x": row["x"], "dup": True}

    pipeline = RefinerPipeline(FakeReader()).map(duplicate_selected)
    payload = compile_pipeline_plan(pipeline)
    steps = payload["stages"][0]["steps"]
    assert steps[1]["name"] == "duplicate_selected"
    assert steps[1]["type"] == "row_map"


def test_compile_pipeline_plan_uses_builtin_calls_for_builtin_steps() -> None:
    pipeline = RefinerPipeline(FakeReader()).map(
        motion_trim(threshold=0.25, pad_frames=2)
    )

    steps = compile_pipeline_plan(pipeline)["stages"][0]["steps"]

    assert steps[1]["name"] == "robotics:motion_trim"
    assert steps[1]["args"] == {
        "action_key": "action",
        "state_key": "observation.state",
        "timestamp_key": "timestamp",
        "threshold": 0.25,
        "pad_frames": 2,
    }


async def _noop_inference(row, generate):
    del generate
    return row


def test_compile_pipeline_plan_includes_runtime_services_for_builtin_steps() -> None:
    pipeline = RefinerPipeline(FakeReader()).map_async(
        rf.inference.generate_text(
            fn=_noop_inference,
            provider=rf.inference.VLLMProvider(model="Qwen/Qwen3.5-9B"),
            default_generation_params={"temperature": 0},
            max_concurrent_requests=7,
        )
    )

    stage = compile_pipeline_plan(pipeline)["stages"][0]
    step = stage["steps"][1]

    assert step["name"] == "inference.generate_text"
    assert step["args"]["fn"] == (
        "async def _noop_inference(row, generate):\n    del generate\n    return row"
    )
    assert step["args"]["provider"] == {
        "type": "vllm",
        "model_name_or_path": "Qwen/Qwen3.5-9B",
        "config": "throughput",
    }
    assert step["args"]["max_concurrent_requests"] == 7
    assert step["args"]["default_generation_params"] == {"temperature": 0}
    assert stage["runtime_services"] == [
        {
            "name": stage["runtime_services"][0]["name"],
            "kind": "llm",
            "config": {
                "model_name_or_path": "Qwen/Qwen3.5-9B",
                "config": "throughput",
            },
        }
    ]
    assert stage["runtime_services"][0]["name"].startswith("vllm-")


def test_compile_pipeline_plan_includes_lerobot_writer_steps() -> None:
    pipeline = (
        RefinerPipeline(FakeReader())
        .map(motion_trim(threshold=0.25, pad_frames=2))
        .write_lerobot("hf://buckets/macrodata/test_bucket/aloha_motion")
    )

    stages = compile_pipeline_plan(pipeline)["stages"]

    assert stages[0]["steps"][1]["name"] == "robotics:motion_trim"
    assert stages[0]["steps"][2]["name"] == "write_lerobot"
    assert stages[0]["steps"][2]["type"] == "writer"
    assert (
        stages[0]["steps"][2]["args"]["path"]
        == "hf://buckets/macrodata/test_bucket/aloha_motion"
    )

    assert [step["name"] for step in stages[1]["steps"]] == [
        "task",
        "write_lerobot_meta_reduce",
    ]
    assert stages[1]["steps"][1]["type"] == "writer"
    assert (
        stages[1]["steps"][1]["args"]["path"]
        == "hf://buckets/macrodata/test_bucket/aloha_motion"
    )


def test_compile_pipeline_plan_includes_jsonl_reducer_steps() -> None:
    pipeline = RefinerPipeline(FakeReader()).write_jsonl("/tmp/output")

    stages = compile_pipeline_plan(pipeline)["stages"]

    assert len(stages) == 2
    assert stages[0]["name"] == "write_jsonl_stage_0"
    assert stages[1]["name"] == "write_jsonl_stage_1"
    assert [step["name"] for step in stages[1]["steps"]] == [
        "task",
        "write_jsonl_reduce",
    ]
    assert stages[1]["steps"][1]["type"] == "writer"
    assert stages[1]["steps"][1]["args"]["path"] == "/tmp/output"
    assert (
        stages[1]["steps"][1]["args"]["filename_template"]
        == "{shard_id}__w{worker_id}.jsonl"
    )


def test_compile_pipeline_plan_includes_sink_without_describe() -> None:
    pipeline = RefinerPipeline(FakeReader(), sink=UndescribedSink())

    steps = compile_pipeline_plan(pipeline)["stages"][0]["steps"]

    assert [step["name"] for step in steps] == ["read_fake", "undescribed"]
    assert steps[1]["type"] == "writer"
    assert steps[1]["index"] == 1
    assert "args" not in steps[1]


def test_compile_pipeline_plan_redacts_sink_callable_args() -> None:
    secret = "md_secret_value"

    class CallableSink(BaseSink):
        def write_block(self, block):
            del block
            return {}, 0

        def describe(self):
            return (
                "callable_sink",
                "writer",
                {"fn": lambda: secret},
            )

    steps = compile_pipeline_plan(
        RefinerPipeline(FakeReader(), sink=CallableSink()),
        secret_values=(secret,),
    )["stages"][0]["steps"]

    assert "md_secret_value" not in steps[1]["args"]["fn"]
    assert steps[1]["args"]["__meta"]["fn"] == "code"


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


def test_plan_pipeline_stages_adds_writer_reducer_stage() -> None:
    pipeline = from_items([{"x": 1}]).write_parquet("/tmp/output")
    stages = plan_pipeline_stages(pipeline, default_num_workers=3)

    assert len(stages) == 2
    assert stages[0].index == 0
    assert stages[0].name == "write_parquet_stage_0"
    assert stages[0].pipeline is pipeline
    assert stages[0].compute.num_workers == 3
    assert stages[1].index == 1
    assert stages[1].name == "write_parquet_stage_1"
    assert stages[1].compute.num_workers == 1
    assert stages[1].compute.inherit_launcher_resources is False
    assert stages[1].pipeline.source.name == "task"


def test_plan_pipeline_sequence_preserves_names_resources_and_order() -> None:
    first = from_items([{"x": 1}])
    second = from_items([{"x": 2}])
    gpu = rf.GPU(count=1, type="h100")
    sequence = first.as_stage(
        name="prepare",
        num_workers=3,
        cpus_per_worker=2,
    ).then(
        second,
        name="publish",
        num_workers=5,
        mem_mb_per_worker=8192,
        gpu=gpu,
    )

    stages = plan_pipeline_stages(sequence, default_num_workers=99)

    assert [stage.index for stage in stages] == [0, 1]
    assert [stage.name for stage in stages] == ["prepare", "publish"]
    assert [stage.pipeline for stage in stages] == [first, second]
    assert stages[0].compute == StageComputeRequirements(
        num_workers=3,
        cpus_per_worker=2,
        inherit_launcher_resources=False,
    )
    assert stages[1].compute == StageComputeRequirements(
        num_workers=5,
        memory_mb_per_worker=8192,
        gpu=gpu,
        inherit_launcher_resources=False,
    )


def test_plan_pipeline_sequence_keeps_generated_finalizer_adjacent() -> None:
    first = from_items([{"x": 1}]).write_parquet("/tmp/output")
    second = from_items([{"x": 2}])
    sequence = first.as_stage(name="prepare", num_workers=3).then(
        second,
        name="publish",
        num_workers=2,
    )

    stages = plan_pipeline_stages(sequence, default_num_workers=99)

    assert [stage.index for stage in stages] == [0, 1, 2]
    assert [stage.name for stage in stages] == [
        "prepare",
        "prepare_finalize",
        "publish",
    ]
    assert [stage.compute.num_workers for stage in stages] == [3, 1, 2]
    assert stages[1].compute.inherit_launcher_resources is False


def test_writer_can_declare_multiple_followup_stages() -> None:
    pipeline = RefinerPipeline(FakeReader(), sink=MultiStageSink())
    sequence = pipeline.as_stage(name="write", num_workers=8)

    stages = plan_pipeline_stages(sequence, default_num_workers=99)

    assert [stage.index for stage in stages] == [0, 1, 2]
    assert [stage.name for stage in stages] == [
        "write",
        "write_index",
        "write_publish",
    ]
    assert [stage.compute.num_workers for stage in stages] == [8, 2, 1]
    assert stages[1].pipeline.source.describe()["num_tasks"] == 2
    assert stages[1].compute.cpus_per_worker == 4
    assert stages[1].compute.inherit_launcher_resources is False
    assert stages[2].compute.inherit_launcher_resources is False


def test_pipeline_sequence_rejects_invalid_or_duplicate_stage_configuration() -> None:
    pipeline = from_items([{"x": 1}])

    with pytest.raises(ValueError, match="stage name must be non-empty"):
        pipeline.as_stage(name=" ")

    sequence = pipeline.as_stage(name="prepare")
    with pytest.raises(ValueError, match="stage names must be unique"):
        sequence.then(pipeline, name="prepare")

    finalizing = (
        pipeline.write_parquet("/tmp/output")
        .as_stage(name="prepare")
        .then(
            pipeline,
            name="prepare_finalize",
        )
    )
    with pytest.raises(ValueError, match="writer follow-up stage"):
        plan_pipeline_stages(finalizing, default_num_workers=1)


def test_pipeline_sequence_preserves_auto_worker_count() -> None:
    sequence = from_items([{"x": 1}]).as_stage(
        name="prepare",
        num_workers="auto",
    )

    stages = plan_pipeline_stages(sequence, default_num_workers=1)

    assert stages[0].compute.num_workers == "auto"
