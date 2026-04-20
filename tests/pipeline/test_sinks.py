from __future__ import annotations

import json
from typing import cast

import pyarrow.parquet as pq
import pytest

from refiner import col
from refiner.pipeline.data.row import DictRow
from refiner.pipeline import from_items
from refiner.pipeline.sinks import JsonlSink
from refiner.pipeline.sinks.parquet import ParquetSink
from refiner.pipeline.sinks.reducer.file import FileCleanupReducerSink
from refiner.worker.context import set_active_run_context
from refiner.worker.lifecycle import FinalizedShardWorker, RuntimeLifecycle
from refiner.worker.context import worker_token_for


class _FinalizedWorkersRuntime:
    def __init__(self, rows: list[FinalizedShardWorker]) -> None:
        self._rows = rows

    def finalized_workers(
        self, *, stage_index: int | None = None
    ) -> list[FinalizedShardWorker]:
        assert stage_index == 0
        return self._rows


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

    assert stats.claimed == 3
    assert stats.completed == 3
    assert stats.output_rows == 3
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

    assert stats.claimed == 3
    assert stats.completed == 3
    assert stats.output_rows == 3
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

    assert stats.claimed == 3
    assert stats.completed == 3
    assert stats.output_rows == 2
    written = sorted(path for path in output_dir.iterdir() if path.suffix == ".jsonl")
    assert len(written) == 2
    assert all("__w" in path.name for path in written)


def test_jsonl_sink_uses_local_worker_suffix_outside_runtime(tmp_path) -> None:
    sink = JsonlSink(tmp_path)
    sink.write_block([DictRow({"x": 1}, shard_id="abc")])
    sink.on_shard_complete("abc")

    written = sorted(tmp_path.iterdir())
    assert [path.name for path in written] == [
        f"abc__w{worker_token_for('local')}.jsonl"
    ]
    assert json.loads(written[0].read_text(encoding="utf-8")) == {"x": 1}


def test_jsonl_reducer_keeps_only_finalized_worker_outputs(tmp_path) -> None:
    output_dir = tmp_path / "jsonl-cleanup"
    shard_id = "0123456789ab"
    worker_ids = ["worker-1", "worker-2"]

    for worker_id, value in zip(worker_ids, [1, 9], strict=True):
        sink = JsonlSink(output_dir)
        with set_active_run_context(
            job_id="job",
            stage_index=0,
            worker_id=worker_id,
            worker_name=None,
            runtime_lifecycle=cast(
                RuntimeLifecycle,
                _FinalizedWorkersRuntime(
                    [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_ids[1])]
                ),
            ),
        ):
            sink.write_block([DictRow({"x": value}, shard_id=shard_id)])
            sink.on_shard_complete(shard_id)

    reducer = JsonlSink(output_dir).build_reducer()
    assert reducer is not None
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(
            RuntimeLifecycle,
            _FinalizedWorkersRuntime(
                [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_ids[1])]
            ),
        ),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    kept = output_dir / f"{shard_id}__w{worker_token_for(worker_ids[1])}.jsonl"
    deleted = output_dir / f"{shard_id}__w{worker_token_for(worker_ids[0])}.jsonl"
    assert kept.exists()
    assert not deleted.exists()
    assert json.loads(kept.read_text(encoding="utf-8")) == {"x": 9}


def test_parquet_reducer_keeps_only_finalized_worker_outputs(tmp_path) -> None:
    output_dir = tmp_path / "parquet-cleanup"
    shard_id = "0123456789ab"
    worker_ids = ["worker-1", "worker-2"]

    for worker_id, value in zip(worker_ids, [1, 9], strict=True):
        sink = ParquetSink(output_dir)
        with set_active_run_context(
            job_id="job",
            stage_index=0,
            worker_id=worker_id,
            worker_name=None,
            runtime_lifecycle=cast(
                RuntimeLifecycle,
                _FinalizedWorkersRuntime(
                    [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_ids[1])]
                ),
            ),
        ):
            sink.write_block([DictRow({"x": value}, shard_id=shard_id)])
            sink.on_shard_complete(shard_id)

    reducer = ParquetSink(output_dir).build_reducer()
    assert reducer is not None
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(
            RuntimeLifecycle,
            _FinalizedWorkersRuntime(
                [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_ids[1])]
            ),
        ),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    kept = output_dir / f"{shard_id}__w{worker_token_for(worker_ids[1])}.parquet"
    deleted = output_dir / f"{shard_id}__w{worker_token_for(worker_ids[0])}.parquet"
    assert kept.exists()
    assert not deleted.exists()
    assert pq.read_table(kept).column("x").to_pylist() == [9]


def test_file_cleanup_reducer_ignores_extra_template_fields(tmp_path) -> None:
    output_dir = tmp_path / "jsonl-cleanup-extra"
    shard_id = "0123456789ab"
    winner_worker_id = "worker-2"
    loser_worker_id = "worker-1"
    winner_token = worker_token_for(winner_worker_id)
    loser_token = worker_token_for(loser_worker_id)

    winner_files = [
        output_dir / f"{shard_id}__w{winner_token}__part0.jsonl",
        output_dir / f"{shard_id}__w{winner_token}__part1.jsonl",
    ]
    loser_file = output_dir / f"{shard_id}__w{loser_token}__part0.jsonl"
    unmanaged_file = output_dir / "notes.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in winner_files + [loser_file]:
        path.write_text("{}", encoding="utf-8")
    unmanaged_file.write_text("keep me", encoding="utf-8")

    reducer = FileCleanupReducerSink(
        output_dir,
        filename_template="{shard_id}__w{worker_id}__{part}.jsonl",
        reducer_name="cleanup_jsonl",
    )
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(
            RuntimeLifecycle,
            _FinalizedWorkersRuntime(
                [FinalizedShardWorker(shard_id=shard_id, worker_id=winner_worker_id)]
            ),
        ),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    assert all(path.exists() for path in winner_files)
    assert not loser_file.exists()
    assert unmanaged_file.exists()


def test_file_cleanup_reducer_tolerates_duplicate_listed_paths(
    tmp_path, monkeypatch
) -> None:
    output_dir = tmp_path / "jsonl-cleanup-duplicates"
    shard_id = "0123456789ab"
    winner_worker_id = "worker-2"
    loser_worker_id = "worker-1"
    winner_path = (
        output_dir / f"{shard_id}__w{worker_token_for(winner_worker_id)}.jsonl"
    )
    loser_path = output_dir / f"{shard_id}__w{worker_token_for(loser_worker_id)}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    winner_path.write_text("{}", encoding="utf-8")
    loser_path.write_text("{}", encoding="utf-8")

    reducer = FileCleanupReducerSink(
        output_dir,
        filename_template="{shard_id}__w{worker_id}.jsonl",
        reducer_name="cleanup_jsonl",
    )
    monkeypatch.setattr(
        reducer.output,
        "find",
        lambda _: [winner_path.name, winner_path.name, loser_path.name],
    )

    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(
            RuntimeLifecycle,
            _FinalizedWorkersRuntime(
                [FinalizedShardWorker(shard_id=shard_id, worker_id=winner_worker_id)]
            ),
        ),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    assert winner_path.exists()
    assert not loser_path.exists()


def test_jsonl_sink_rejects_unsupported_cleanup_filename_template(tmp_path) -> None:
    sink = JsonlSink(
        tmp_path / "jsonl-custom",
        filename_template="{shard_id}.jsonl",
    )

    with pytest.raises(ValueError, match="requires fields"):
        sink.build_reducer()


def test_parquet_sink_rejects_unsupported_cleanup_filename_template(tmp_path) -> None:
    sink = ParquetSink(
        tmp_path / "parquet-custom",
        filename_template="{shard_id:>12}.parquet",
    )

    with pytest.raises(ValueError, match="without conversion or format specifiers"):
        sink.build_reducer()
