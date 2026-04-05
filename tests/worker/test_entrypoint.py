from __future__ import annotations

import cloudpickle
import sys

from refiner.platform.client.http import MacrodataApiError
from refiner.worker import entrypoint


def test_entrypoint_exits_cleanly_when_stage_is_already_terminal(
    monkeypatch, tmp_path, capsys
) -> None:
    payload_path = tmp_path / "pipeline.cloudpickle"
    payload_path.write_bytes(cloudpickle.dumps(object()))

    class _FailingClient:
        def __init__(self) -> None:
            self.base_url = "http://localhost"
            self.api_key = "md_test"

        def report_worker_started(self, **kwargs):
            del kwargs
            raise MacrodataApiError(
                status=409,
                message="Cannot start worker for stage 0 in terminal state failed",
            )

    monkeypatch.setattr(entrypoint, "MacrodataClient", lambda: _FailingClient())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refiner.worker.entrypoint",
            "--pipeline-payload",
            str(payload_path),
            "--job-id",
            "job-1",
            "--stage-index",
            "0",
            "--runtime-backend",
            "platform",
        ],
    )

    assert entrypoint.main() == 0
    assert (
        '"skipped": "HTTP 409: Cannot start worker for stage 0 in terminal state failed"'
        in (capsys.readouterr().out)
    )


def test_entrypoint_sets_visible_gpu_ids(monkeypatch, tmp_path) -> None:
    payload_path = tmp_path / "pipeline.cloudpickle"
    payload_path.write_bytes(cloudpickle.dumps(object()))
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        entrypoint, "MacrodataClient", lambda: (_ for _ in ()).throw(AssertionError())
    )
    monkeypatch.setattr(entrypoint, "set_cpu_affinity", lambda cpu_ids: None)
    monkeypatch.setattr(
        entrypoint,
        "set_visible_gpu_ids",
        lambda gpu_ids: seen.setdefault("gpu_ids", gpu_ids),
    )

    class _FakeWorker:
        def __init__(self, **kwargs):
            del kwargs

        def run(self):
            class _Stats:
                claimed = 0
                completed = 0
                failed = 0
                output_rows = 0

            return _Stats()

    monkeypatch.setattr(entrypoint, "Worker", _FakeWorker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refiner.worker.entrypoint",
            "--pipeline-payload",
            str(payload_path),
            "--job-id",
            "job-1",
            "--stage-index",
            "0",
            "--runtime-backend",
            "file",
            "--gpu-ids",
            "0,3",
        ],
    )

    assert entrypoint.main() == 0
    assert seen["gpu_ids"] == ["0", "3"]


def test_entrypoint_passes_preregistered_worker_id_to_worker(
    monkeypatch, tmp_path
) -> None:
    payload_path = tmp_path / "pipeline.cloudpickle"
    payload_path.write_bytes(cloudpickle.dumps(object()))
    seen: dict[str, object] = {}

    class _FakeClient:
        def __init__(self) -> None:
            self.base_url = "http://localhost"
            self.api_key = "md_test"

    class _FakeWorker:
        def __init__(self, **kwargs):
            seen["run_handle"] = kwargs["run_handle"]

        def run(self):
            class _Stats:
                claimed = 0
                completed = 0
                failed = 0
                output_rows = 0

            return _Stats()

    monkeypatch.setattr(entrypoint, "MacrodataClient", _FakeClient)
    monkeypatch.setattr(entrypoint, "Worker", _FakeWorker)
    monkeypatch.setattr(entrypoint, "set_cpu_affinity", lambda cpu_ids: None)
    monkeypatch.setattr(entrypoint, "set_visible_gpu_ids", lambda gpu_ids: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refiner.worker.entrypoint",
            "--pipeline-payload",
            str(payload_path),
            "--job-id",
            "job-1",
            "--stage-index",
            "0",
            "--runtime-backend",
            "platform",
            "--worker-id",
            "worker-pre",
        ],
    )

    assert entrypoint.main() == 0
    run_handle = seen["run_handle"]
    assert isinstance(run_handle, entrypoint.RunHandle)
    assert run_handle.worker_id == "worker-pre"


def test_entrypoint_passes_service_control_url(monkeypatch, tmp_path) -> None:
    payload_path = tmp_path / "pipeline.cloudpickle"
    payload_path.write_bytes(cloudpickle.dumps(object()))
    seen: dict[str, object] = {}

    class _FakeWorker:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def run(self):
            class _Stats:
                claimed = 0
                completed = 0
                failed = 0
                output_rows = 0

            return _Stats()

    monkeypatch.setattr(
        entrypoint, "MacrodataClient", lambda: (_ for _ in ()).throw(AssertionError())
    )
    monkeypatch.setattr(entrypoint, "Worker", _FakeWorker)
    monkeypatch.setattr(entrypoint, "set_cpu_affinity", lambda cpu_ids: None)
    monkeypatch.setattr(entrypoint, "set_visible_gpu_ids", lambda gpu_ids: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refiner.worker.entrypoint",
            "--pipeline-payload",
            str(payload_path),
            "--job-id",
            "job-1",
            "--stage-index",
            "0",
            "--runtime-backend",
            "file",
            "--service-control-url",
            "http://127.0.0.1:9123",
        ],
    )

    assert entrypoint.main() == 0
    assert seen["service_control_url"] == "http://127.0.0.1:9123"
