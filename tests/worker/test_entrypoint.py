from __future__ import annotations

from refiner.worker import entrypoint
from refiner.worker.context import _base_logger
from refiner.worker.metrics.emitter import LocalLogEmitter


def test_entrypoint_sets_visible_gpus_before_loading_pipeline(
    tmp_path, monkeypatch
) -> None:
    payload_path = tmp_path / "pipeline.cloudpickle"
    payload_path.write_bytes(b"placeholder")
    assignments_dir = tmp_path / "stage-2" / "assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)
    (assignments_dir / "worker-worker-1.json").write_text("[]")

    events: list[str] = []

    monkeypatch.setattr(
        "sys.argv",
        [
            "entrypoint.py",
            "--pipeline-payload",
            str(payload_path),
            "--job-id",
            "job-1",
            "--stage-index",
            "2",
            "--worker-name",
            "worker-name",
            "--worker-id",
            "worker-1",
            "--rundir",
            str(tmp_path),
            "--gpu-ids",
            "0,1",
        ],
    )
    monkeypatch.setattr(
        entrypoint,
        "set_visible_gpu_ids",
        lambda gpu_ids: events.append(f"set:{','.join(gpu_ids)}"),
    )
    monkeypatch.setattr(
        entrypoint.cloudpickle,
        "load",
        lambda handle: events.append("load") or object(),
    )

    def fake_worker(**kwargs):
        return type(
            "_FakeWorker",
            (),
            {
                "run": staticmethod(
                    lambda: type(
                        "_FakeStats",
                        (),
                        {
                            "claimed": 0,
                            "completed": 0,
                            "failed": 0,
                            "output_rows": 0,
                        },
                    )()
                )
            },
        )()

    monkeypatch.setattr(entrypoint, "Worker", fake_worker)

    assert entrypoint.main() == 0
    assert events[:2] == ["set:0,1", "load"]


def test_entrypoint_adds_and_removes_local_log_sink(tmp_path, monkeypatch) -> None:
    payload_path = tmp_path / "pipeline.cloudpickle"
    payload_path.write_bytes(b"placeholder")
    assignments_dir = tmp_path / "stage-0" / "assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)
    (assignments_dir / "worker-worker-1.json").write_text("[]")
    log_path = tmp_path / "stage-0" / "logs" / "worker-worker-1.log"

    events: list[str] = []

    monkeypatch.setattr(
        "sys.argv",
        [
            "entrypoint.py",
            "--pipeline-payload",
            str(payload_path),
            "--job-id",
            "job-1",
            "--stage-index",
            "0",
            "--worker-name",
            "worker-name",
            "--worker-id",
            "worker-1",
            "--rundir",
            str(tmp_path),
        ],
    )

    monkeypatch.setattr(
        _base_logger,
        "add",
        lambda path, enqueue=False, catch=False: events.append(f"add:{path}") or 123,
    )
    monkeypatch.setattr(_base_logger, "complete", lambda: events.append("complete"))
    monkeypatch.setattr(
        _base_logger,
        "remove",
        lambda sink_id: events.append(f"remove:{sink_id}"),
    )
    monkeypatch.setattr(entrypoint.cloudpickle, "load", lambda handle: object())

    def fake_worker(**kwargs):
        return type(
            "_FakeWorker",
            (),
            {
                "run": staticmethod(
                    lambda: type(
                        "_FakeStats",
                        (),
                        {
                            "claimed": 0,
                            "completed": 0,
                            "failed": 0,
                            "output_rows": 0,
                        },
                    )()
                )
            },
        )()

    monkeypatch.setattr(entrypoint, "Worker", fake_worker)

    assert entrypoint.main() == 0
    assert events == [f"add:{log_path}", "complete", "remove:123"]


def test_entrypoint_passes_local_log_emitter_to_worker(tmp_path, monkeypatch) -> None:
    payload_path = tmp_path / "pipeline.cloudpickle"
    payload_path.write_bytes(b"placeholder")
    assignments_dir = tmp_path / "stage-0" / "assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)
    (assignments_dir / "worker-worker-1.json").write_text("[]")

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "entrypoint.py",
            "--pipeline-payload",
            str(payload_path),
            "--job-id",
            "job-1",
            "--stage-index",
            "0",
            "--worker-name",
            "worker-name",
            "--worker-id",
            "worker-1",
            "--rundir",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(entrypoint.cloudpickle, "load", lambda handle: object())
    monkeypatch.setattr(
        _base_logger,
        "add",
        lambda path, enqueue=False, catch=False: 123,
    )
    monkeypatch.setattr(_base_logger, "remove", lambda sink_id: None)
    monkeypatch.setattr(_base_logger, "complete", lambda: None)

    def fake_worker(**kwargs):
        captured.update(kwargs)
        return type(
            "_FakeWorker",
            (),
            {
                "run": staticmethod(
                    lambda: type(
                        "_FakeStats",
                        (),
                        {
                            "claimed": 0,
                            "completed": 0,
                            "failed": 0,
                            "output_rows": 0,
                        },
                    )()
                )
            },
        )()

    monkeypatch.setattr(entrypoint, "Worker", fake_worker)

    assert entrypoint.main() == 0
    assert isinstance(captured["user_metrics_emitter"], LocalLogEmitter)
