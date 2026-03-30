from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    cpu_cores: int | None = None
    memory_mb: int | None = None
    gpu_count: int | None = None
    gpu_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.cpu_cores is not None:
            payload["cpu_cores"] = self.cpu_cores
        if self.memory_mb is not None:
            payload["memory_mb"] = self.memory_mb
        if self.gpu_count is not None:
            payload["gpu_count"] = self.gpu_count
        if self.gpu_type is not None:
            payload["gpu_type"] = self.gpu_type
        return payload


__all__ = ["WorkerConfig"]
