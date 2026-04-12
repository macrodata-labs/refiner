from __future__ import annotations

import cloudpickle
import os
import sys
from refiner.platform.client.api import MacrodataApiError
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
        ],
    )

    assert entrypoint.main() == 0
    assert (
        '"skipped": "HTTP 409: Cannot start worker for stage 0 in terminal state failed"'
        in (capsys.readouterr().out)
    )


def test_entrypoint_sets_refiner_workdir_env(monkeypatch, tmp_path) -> None:
    payload_path = tmp_path / "pipeline.cloudpickle"
    payload_path.write_bytes(cloudpickle.dumps(object()))

    class _Started:
        worker_id = "worker-1"

    class _Client:
        def __init__(self) -> None:
            self.base_url = "http://localhost"
            self.api_key = "md_test"

        def report_worker_started(self, **kwargs):
            del kwargs
            return _Started()

    class _FakeWorker:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def run(self):
            class _Stats:
                claimed = 0
                completed = 0
                failed = 0
                output_rows = 0

            return _Stats()

    monkeypatch.delenv("REFINER_WORKDIR", raising=False)
    monkeypatch.setattr(entrypoint, "MacrodataClient", lambda: _Client())
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
            "--workdir",
            str(tmp_path / "workdir"),
        ],
    )

    assert entrypoint.main() == 0
    assert os.environ["REFINER_WORKDIR"] == str(tmp_path / "workdir")
