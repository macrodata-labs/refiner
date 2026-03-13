from __future__ import annotations

from collections.abc import Iterator

import pytest

from refiner.pipeline.data.shard import Shard
from refiner.pipeline import RefinerPipeline, read_jsonl
from refiner.launchers.local import LocalLauncher
from refiner.pipeline.planning import PlannedStage, StageComputeRequirements
from refiner.pipeline.sources.readers.base import BaseReader
from refiner.pipeline.data.row import DictRow, Row
from refiner.worker.resources.cpu import build_cpu_sets


class _FakeReader(BaseReader):
    def __init__(self, shards: list[Shard], rows_by_shard_id: dict[str, list[Row]]):
        super().__init__(inputs=[])
        self._shards = shards
        self._rows_by_shard_id = rows_by_shard_id

    def list_shards(self) -> list[Shard]:
        return list(self._shards)

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        yield from self._rows_by_shard_id.get(shard.id, [])


def test_launch_local_single_worker(tmp_path) -> None:
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text('{"x": 1}\n{"x": 2}\n')
    p2.write_text('{"x": 3}\n')

    pipeline = (
        read_jsonl([str(p1), str(p2)])
        .map(lambda r: {"x": int(r["x"]) + 1})
        .filter(lambda r: int(r["x"]) % 2 == 0)
    )

    stats = pipeline.launch_local(
        name="unit-test-local",
        num_workers=1,
        workdir=str(tmp_path),
    )

    assert stats.workers == 1
    assert stats.claimed == 2
    assert stats.completed == 2
    assert stats.failed == 0
    assert stats.output_rows == 2


def test_build_cpu_sets_partitions_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "refiner.worker.resources.cpu.available_cpu_ids",
        lambda: [0, 1, 2, 3, 4, 5],
    )
    sets = build_cpu_sets(num_workers=3, cpus_per_worker=2)
    assert sets == [[0, 1], [2, 3], [4, 5]]


def test_build_cpu_sets_raises_when_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "refiner.worker.resources.cpu.available_cpu_ids",
        lambda: [0, 1, 2],
    )
    with pytest.raises(ValueError):
        build_cpu_sets(num_workers=2, cpus_per_worker=2)


def test_launch_local_multi_worker_subprocess_with_lambda(tmp_path) -> None:
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text('{"x": 1}\n')
    p2.write_text('{"x": 2}\n')
    pipeline = read_jsonl([str(p1), str(p2)]).map(lambda r: {"x": int(r["x"]) + 10})

    stats = pipeline.launch_local(
        name="unit-test-local-subprocess",
        num_workers=2,
        workdir=str(tmp_path),
    )
    assert stats.workers == 2
    assert stats.claimed == 2
    assert stats.completed == 2
    assert stats.failed == 0
    assert stats.output_rows == 2


def test_local_launcher_file_backend_skips_platform_setup(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))

    launcher = LocalLauncher(
        pipeline=pipeline,
        name="file-backend-no-platform",
        num_workers=1,
        workdir=str(tmp_path),
        runtime_backend="file",
    )

    def _unexpected_setup(**kwargs):  # noqa: ANN003
        del kwargs
        raise AssertionError("_setup_platform should not be called in file mode")

    launcher._setup_platform = _unexpected_setup  # type: ignore[method-assign]
    stats = launcher.launch()
    assert stats.completed == 1


def test_local_launcher_platform_backend_requires_platform_client(monkeypatch) -> None:
    shard = Shard(path="a", start=0, end=1)
    rows = {shard.id: [DictRow({"x": 1})]}
    pipeline = RefinerPipeline(source=_FakeReader([shard], rows))

    launcher = LocalLauncher(
        pipeline=pipeline,
        name="platform-backend-required",
        num_workers=1,
        runtime_backend="platform",
    )

    monkeypatch.setattr(launcher, "_platform_client_or_none", lambda: None)
    with pytest.raises(
        RuntimeError, match="platform runtime requires Macrodata authentication"
    ):
        launcher.launch()


def test_launch_local_runs_planned_stages_sequentially(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_path = tmp_path / "stage0.jsonl"
    second_path = tmp_path / "stage1.jsonl"
    first_path.write_text('{"x": 1}\n{"x": 2}\n')
    second_path.write_text('{"x": 3}\n')

    pipeline = read_jsonl(str(first_path))
    stage_zero = read_jsonl(str(first_path))
    stage_one = read_jsonl(str(second_path)).map(lambda row: {"x": int(row["x"]) + 10})

    monkeypatch.setattr(
        "refiner.launchers.base.plan_pipeline_stages",
        lambda pipeline, default_num_workers: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=stage_zero,
                compute=StageComputeRequirements(num_workers=1),
            ),
            PlannedStage(
                index=1,
                name="stage_1",
                pipeline=stage_one,
                compute=StageComputeRequirements(num_workers=2),
            ),
        ],
    )

    stats = pipeline.launch_local(
        name="unit-test-local-multi-stage",
        num_workers=4,
        workdir=str(tmp_path),
        runtime_backend="file",
    )

    assert stats.workers == 3
    assert stats.claimed == 2
    assert stats.completed == 2
    assert stats.failed == 0
    assert stats.output_rows == 3
    assert (tmp_path / "runs" / stats.job_id / "launcher" / "stage-0").exists()
    assert (tmp_path / "runs" / stats.job_id / "launcher" / "stage-1").exists()
