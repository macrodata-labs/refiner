from __future__ import annotations
import json

from refiner.services import VLLMRuntimeServiceBinding
from refiner.worker.service_control import request_runtime_service_bindings


def test_request_runtime_service_bindings_posts_services_and_parses_response(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "services": [
                        {
                            "name": "vllm-test",
                            "kind": "llm",
                            "endpoint": "http://127.0.0.1:8000",
                        }
                    ]
                },
                sort_keys=True,
            ).encode("utf-8")

    def fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    bindings = request_runtime_service_bindings(
        control_url="http://127.0.0.1:9123",
        worker_id="worker-0",
    )

    assert bindings == (
        VLLMRuntimeServiceBinding(
            name="vllm-test",
            kind="llm",
            endpoint="http://127.0.0.1:8000",
        ),
    )
    assert captured["url"] == "http://127.0.0.1:9123/services/start"
    assert captured["timeout"] == 600.0
    assert captured["body"] == {
        "worker_id": "worker-0",
    }
