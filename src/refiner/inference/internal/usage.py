from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from refiner.inference.internal.response import InferenceResponse
from refiner.pipeline.data.row import Row


def record_usage(row: Row, response: Any) -> None:
    if not isinstance(response, InferenceResponse):
        return
    row.log_throughput(
        "prompt_tokens", _usage_int(response.usage, "prompt_tokens"), unit="tokens"
    )
    row.log_throughput(
        "completion_tokens",
        _usage_int(response.usage, "completion_tokens"),
        unit="tokens",
    )


def _usage_int(usage: Mapping[str, Any], key: str) -> int:
    raw = usage.get(key, 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


__all__ = ["record_usage"]
