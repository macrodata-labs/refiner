from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CloudRuntimeConfig:
    num_workers: int
    heartbeat_every_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "heartbeat_every_rows": self.heartbeat_every_rows,
        }


@dataclass(frozen=True, slots=True)
class CloudPipelinePayload:
    format: str
    bytes_b64: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "bytes_b64": self.bytes_b64,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class CloudRunCreateRequest:
    name: str
    plan: dict[str, Any]
    runtime: CloudRuntimeConfig
    pipeline_payload: CloudPipelinePayload

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "executor": {"type": "macrodata-cloud"},
            "plan": self.plan,
            "runtime": self.runtime.to_dict(),
            "pipeline_payload": self.pipeline_payload.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CloudRunCreateResponse:
    job_id: str
    stage_id: str
    status: str
