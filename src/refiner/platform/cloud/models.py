from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CloudRuntimeConfig:
    num_workers: int
    heartbeat_every_rows: int
    cpus_per_worker: int | None = None
    mem_mb_per_worker: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "num_workers": self.num_workers,
            "heartbeat_every_rows": self.heartbeat_every_rows,
        }
        if self.cpus_per_worker is not None:
            payload["cpus_per_worker"] = self.cpus_per_worker
        if self.mem_mb_per_worker is not None:
            payload["mem_mb_per_worker"] = self.mem_mb_per_worker
        return payload


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
    shards: list[dict[str, Any]]
    manifest: dict[str, Any] | None = None
    sync_local_dependencies: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "executor": {
                "type": "macrodata-cloud",
                "sync_local_dependencies": self.sync_local_dependencies,
            },
            "plan": self.plan,
            "runtime": self.runtime.to_dict(),
            "pipeline_payload": self.pipeline_payload.to_dict(),
            "shards": self.shards,
        }
        if self.manifest is not None:
            payload["manifest"] = self.manifest
        return payload


@dataclass(frozen=True, slots=True)
class CloudRunCreateResponse:
    job_id: str
    stage_id: str
    status: str
