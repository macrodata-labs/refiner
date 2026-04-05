from __future__ import annotations

import json
import urllib.error
import urllib.request

from refiner.services import RuntimeServiceBinding, parse_runtime_service_bindings


def request_runtime_service_bindings(
    *,
    control_url: str,
    worker_id: str,
    timeout_seconds: float = 600.0,
) -> tuple[RuntimeServiceBinding, ...]:
    if not worker_id.strip():
        raise ValueError("worker_id is required to start runtime services")
    request_body = json.dumps(
        {
            "worker_id": worker_id,
        },
        sort_keys=True,
    ).encode("utf-8")
    request = urllib.request.Request(
        url=f"{control_url.rstrip('/')}/services/start",
        data=request_body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        message = detail or str(exc)
        raise RuntimeError(
            f"runtime services control request failed: {message}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"runtime services control request failed: {exc.reason}"
        ) from exc
    payload = json.loads(response_body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("runtime services control response must be a JSON object")
    return parse_runtime_service_bindings(payload)
