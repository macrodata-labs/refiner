from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import pytest

from refiner.pipeline.data.shard import FilePart, Shard
from refiner.pipeline import RefinerPipeline, read_csv, read_jsonl
from refiner.launchers.local import LaunchStats, LocalLauncher
from refiner.pipeline.planning import PlannedStage, StageComputeRequirements
from refiner.pipeline.sources.readers.base import BaseReader
from refiner.pipeline.data.row import DictRow, Row
from refiner.platform.auth import MacrodataCredentialsError
from refiner.worker.resources.gpu import build_gpu_sets


@pytest.fixture(autouse=True)
def _disable_local_init_api_ping(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("refiner.launchers.local.request_json", lambda **kwargs: {})
    monkeypatch.setenv("REFINER_WORKDIR", str(tmp_path))


class _FakeReader(BaseReader):
    def __init__(
        self,
        shards: list[Shard],
        rows_by_shard_id: Mapping[str, Sequence[Row]],
    ):
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
        rundir=str(tmp_path / "run"),
    )

    assert stats.workers == 1
    assert stats.claimed == 1
    assert stats.completed == 1
    assert stats.failed == 0
    assert stats.output_rows == 2


def test_launch_local_single_worker_csv(tmp_path) -> None:
    path = tmp_path / "a.csv"
    path.write_text("x\n1\n2\n")

    pipeline = (
        read_csv(str(path))
        .map(lambda r: {"x": int(r["x"]) + 1})
        .filter(lambda r: int(r["x"]) % 2 == 0)
    )

    stats = pipeline.launch_local(
        name="unit-test-local-csv",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    assert stats.workers == 1
    assert stats.claimed == 1
    assert stats.completed == 1
    assert stats.failed == 0
    assert stats.output_rows == 1


def test_launch_local_coalesces_writer_shards(tmp_path) -> None:
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text('{"x": 1}\n')
    p2.write_text('{"x": 2}\n')

    pipeline = read_jsonl([str(p1), str(p2)], num_shards=1).write_jsonl(
        tmp_path / "out"
    )

    stats = pipeline.launch_local(
        name="unit-test-local-coalesced",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    assert stats.claimed == 1
    assert stats.completed == 1


def test_build_gpu_sets_partitions_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "refiner.worker.resources.gpu.available_gpu_ids",
        lambda: ["0", "1", "2", "3"],
    )
    sets = build_gpu_sets(num_workers=2, gpus_per_worker=2)
    assert sets == [["0", "1"], ["2", "3"]]


def test_build_gpu_sets_raises_when_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "refiner.worker.resources.gpu.available_gpu_ids",
        lambda: ["0"],
    )
    with pytest.raises(ValueError):
        build_gpu_sets(num_workers=2, gpus_per_worker=1)


def test_launch_local_assigns_visible_gpus(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))

    monkeypatch.setattr(
        "refiner.worker.resources.gpu.available_gpu_ids",
        lambda: ["0"],
    )

    stats = pipeline.launch_local(
        name="local-gpu-launch",
        num_workers=1,
        gpus_per_worker=1,
        rundir=str(tmp_path / "run"),
    )

    assert stats.workers == 1
    assert stats.completed == 1
    assert stats.failed == 0


def test_launch_local_multi_worker_subprocess_with_lambda(tmp_path) -> None:
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text('{"x": 1}\n')
    p2.write_text('{"x": 2}\n')
    pipeline = read_jsonl([str(p1), str(p2)]).map(lambda r: {"x": int(r["x"]) + 10})

    stats = pipeline.launch_local(
        name="unit-test-local-subprocess",
        num_workers=2,
        rundir=str(tmp_path / "run"),
    )
    assert stats.workers == 2
    assert stats.claimed == 1
    assert stats.completed == 1
    assert stats.failed == 0
    assert stats.output_rows == 2


def test_launch_local_ignores_non_json_stdout_before_final_stats(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')

    def noisy_map(row):
        print(f"processing {row['x']}")
        return {"x": int(row["x"]) + 1}

    pipeline = read_jsonl(str(path)).map(noisy_map)

    stats = pipeline.launch_local(
        name="unit-test-local-noisy-stdout",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    assert stats.workers == 1
    assert stats.completed == 1
    assert stats.failed == 0
    assert stats.output_rows == 1


def test_local_launcher_registers_job_and_reports_stage_lifecycle(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))
    create_calls: list[dict[str, object]] = []
    started: list[tuple[str, int]] = []
    finished: list[tuple[str, int, str]] = []

    class _FakeClient:
        base_url = "https://macrodata.co"

        def create_job(self, **kwargs):
            create_calls.append(kwargs)
            from refiner.platform.client.models import CreateJobResponse

            return CreateJobResponse(
                job_id="job-remote",
                stage_index=0,
                workspace_slug="workspace",
            )

        def report_stage_started(self, *, job_id: str, stage_index: int):
            started.append((job_id, stage_index))

        def report_stage_finished(self, *, job_id: str, stage_index: int, status: str):
            finished.append((job_id, stage_index, status))

    monkeypatch.setattr("refiner.launchers.local.current_api_key", lambda: "md_test")
    monkeypatch.setattr(
        "refiner.launchers.local.MacrodataClient", lambda **kwargs: _FakeClient()
    )

    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-launch-register",
        num_workers=1,
    )
    monkeypatch.setattr(
        launcher,
        "_planned_stages",
        lambda: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=pipeline,
                compute=StageComputeRequirements(num_workers=1),
            )
        ],
    )
    monkeypatch.setattr(
        launcher,
        "_launch_stage",
        lambda *, stage: LaunchStats(
            job_id="job-remote",
            workers=1,
            claimed=1,
            completed=1,
            failed=0,
            output_rows=1,
        ),
    )
    launcher.launch()

    assert create_calls
    assert create_calls[0]["executor"] == {"type": "refiner-local"}
    assert started == [("job-remote", 0)]
    assert finished == [("job-remote", 0, "completed")]


def test_local_launcher_warns_without_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    shard = Shard.from_file_parts([FilePart(path="a", start=0, end=1)])
    rows = {shard.id: [DictRow({"x": 1})]}
    pipeline = RefinerPipeline(source=_FakeReader([shard], rows))
    warnings: list[str] = []

    monkeypatch.setattr(
        "refiner.launchers.local.current_api_key",
        lambda: (_ for _ in ()).throw(
            MacrodataCredentialsError("missing", missing=True)
        ),
    )
    monkeypatch.setattr(
        "refiner.launchers.local.logger.warning",
        lambda message: warnings.append(message),
    )

    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-launch-no-creds",
        num_workers=1,
    )
    monkeypatch.setattr(launcher, "_planned_stages", lambda: [])
    launcher.launch()

    assert warnings == [
        "No valid Macrodata API key found. Run `macrodata login` to track local jobs."
    ]


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

    rundir = tmp_path / "run"
    stats = pipeline.launch_local(
        name="unit-test-local-multi-stage",
        num_workers=4,
        rundir=str(rundir),
    )

    assert stats.workers == 3
    assert stats.claimed == 2
    assert stats.completed == 2
    assert stats.failed == 0
    assert stats.output_rows == 3
    assert (rundir / "stage-0").exists()
    assert (rundir / "stage-1").exists()


def test_launch_local_uses_explicit_rundir(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))
    rundir = tmp_path / "custom-run"

    stats = pipeline.launch_local(
        name="unit-test-local-rundir",
        num_workers=1,
        rundir=str(rundir),
    )

    assert (rundir / "stage-0").exists()
    assert not (tmp_path / "runs" / stats.job_id).exists()


def test_launch_local_resumes_from_existing_rundir(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n{"x": 2}\n')
    pipeline = read_jsonl(str(path), num_shards=2)
    rundir = tmp_path / "resume-run"
    stage_manifest = rundir / "stage-0" / "completed" / "worker-1.jsonl"
    first_shard = pipeline.list_shards()[0]
    stage_manifest.parent.mkdir(parents=True, exist_ok=True)
    stage_manifest.write_text(f'{{"shard_id": "{first_shard.id}"}}\n')

    stats = pipeline.launch_local(
        name="unit-test-local-resume",
        num_workers=2,
        rundir=str(rundir),
    )

    assert stats.claimed == 1
    assert stats.completed == 1


def test_local_launcher_stops_after_failed_stage(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = read_jsonl(str(tmp_path / "missing.jsonl"))
    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-stop-on-failure",
        num_workers=1,
    )

    stages = [
        PlannedStage(
            index=0,
            name="stage_0",
            pipeline=pipeline,
            compute=StageComputeRequirements(num_workers=1),
        ),
        PlannedStage(
            index=1,
            name="stage_1",
            pipeline=pipeline,
            compute=StageComputeRequirements(num_workers=1),
        ),
    ]
    launched: list[int] = []

    monkeypatch.setattr(launcher, "_planned_stages", lambda: stages)

    def fake_launch_stage(*, stage):  # noqa: ANN001
        launched.append(stage.index)
        return LaunchStats(
            job_id="job-1",
            workers=1,
            claimed=1,
            completed=0,
            failed=1 if stage.index == 0 else 0,
            output_rows=0,
        )

    monkeypatch.setattr(launcher, "_launch_stage", fake_launch_stage)

    with pytest.raises(RuntimeError, match=r"stage 0 failed"):
        launcher.launch()

    assert launched == [0]


def test_local_launcher_does_not_force_platform_terminal_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))
    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-no-forced-platform-finish",
        num_workers=1,
    )

    monkeypatch.setattr(
        launcher,
        "_planned_stages",
        lambda: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=pipeline,
                compute=StageComputeRequirements(num_workers=1),
            )
        ],
    )

    monkeypatch.setattr(
        launcher,
        "_launch_stage",
        lambda *, stage: LaunchStats(
            job_id="job-1",
            workers=1,
            claimed=1,
            completed=1,
            failed=0,
            output_rows=1,
        ),
    )
    stats = launcher.launch()

    assert stats.completed == 1


def test_local_launcher_reports_failed_stage_to_tracking(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))
    finished: list[tuple[str, int, str]] = []

    class _FakeClient:
        base_url = "https://macrodata.co"

        def create_job(self, **kwargs):
            from refiner.platform.client.models import CreateJobResponse

            return CreateJobResponse(
                job_id="job-remote",
                stage_index=0,
                workspace_slug="workspace",
            )

        def report_stage_started(self, *, job_id: str, stage_index: int):
            return None

        def report_stage_finished(self, *, job_id: str, stage_index: int, status: str):
            finished.append((job_id, stage_index, status))

    monkeypatch.setattr("refiner.launchers.local.current_api_key", lambda: "md_test")
    monkeypatch.setattr(
        "refiner.launchers.local.MacrodataClient", lambda **kwargs: _FakeClient()
    )

    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-stage-fail-report",
        num_workers=1,
    )
    monkeypatch.setattr(
        launcher,
        "_planned_stages",
        lambda: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=pipeline,
                compute=StageComputeRequirements(num_workers=1),
            )
        ],
    )
    monkeypatch.setattr(
        launcher,
        "_launch_stage",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        launcher.launch()

    assert finished == [("job-remote", 0, "failed")]
