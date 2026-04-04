from __future__ import annotations

import cloudpickle
import json
import sys
from refiner.platform.client.http import MacrodataApiError
from refiner.services import RuntimeServiceBinding
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


def test_entrypoint_loads_service_bindings_and_passes_them_to_worker(
    monkeypatch, tmp_path
) -> None:
    payload_path = tmp_path / "pipeline.cloudpickle"
    payload_path.write_bytes(cloudpickle.dumps(object()))
    bindings_path = tmp_path / "bindings.json"
    bindings_path.write_text(
        json.dumps(
            {
                "services": [
                    {
                        "name": "llm",
                        "kind": "llm",
                        "endpoint": "http://127.0.0.1:9000/v1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        entrypoint, "MacrodataClient", lambda: (_ for _ in ()).throw(AssertionError())
    )

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
            "--service-bindings-path",
            str(bindings_path),
        ],
    )

    assert entrypoint.main() == 0
    service_bindings = seen["service_bindings"]
    assert isinstance(service_bindings, tuple)
    assert len(service_bindings) == 1
    binding = service_bindings[0]
    assert isinstance(binding, RuntimeServiceBinding)
    assert binding.name == "llm"
