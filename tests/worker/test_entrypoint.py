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
