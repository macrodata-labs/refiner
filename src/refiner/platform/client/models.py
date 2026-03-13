from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import msgspec

from refiner.pipeline.data.shard import Shard


class WorkspaceIdentity(msgspec.Struct, frozen=True):
    name: str
    slug: str


class UserIdentity(msgspec.Struct, frozen=True):
    name: str | None = None
    username: str | None = None
    email: str | None = None


class VerifyApiKeyResponse(msgspec.Struct, frozen=True):
    name: str
    key_id: str | None = msgspec.field(name="id", default=None)
    workspace: WorkspaceIdentity | None = None
    user: UserIdentity = msgspec.field(default_factory=UserIdentity)


class StageSummary(msgspec.Struct, frozen=True):
    index: int


class JobSummary(msgspec.Struct, frozen=True):
    id: str
    stages: list[StageSummary]
    workspace_slug: str | None = msgspec.field(name="workspaceSlug", default=None)


class CreateJobEnvelope(msgspec.Struct, frozen=True):
    job: JobSummary


class CreateJobResponse(msgspec.Struct, frozen=True):
    job_id: str
    stage_index: int
    workspace_slug: str | None = None

    @classmethod
    def from_envelope(cls, envelope: CreateJobEnvelope) -> CreateJobResponse:
        if not envelope.job.stages:
            raise ValueError("job submit response missing stages")
        workspace_slug = envelope.job.workspace_slug
        return cls(
            job_id=envelope.job.id,
            stage_index=envelope.job.stages[0].index,
            workspace_slug=workspace_slug.strip()
            if workspace_slug and workspace_slug.strip()
            else None,
        )


@dataclass(frozen=True, slots=True)
class RunHandle:
    job_id: str
    stage_index: int
    client: Any | None = None
    workspace_slug: str | None = None
    worker_name: str | None = None
    worker_id: str | None = None

    def with_worker(
        self,
        *,
        worker_name: str | None = None,
        worker_id: str | None = None,
    ) -> RunHandle:
        return RunHandle(
            job_id=self.job_id,
            stage_index=self.stage_index,
            client=self.client,
            workspace_slug=self.workspace_slug,
            worker_name=worker_name if worker_name is not None else self.worker_name,
            worker_id=worker_id if worker_id is not None else self.worker_id,
        )

    def with_stage(self, stage_index: int) -> RunHandle:
        return RunHandle(
            job_id=self.job_id,
            stage_index=stage_index,
            client=self.client,
            workspace_slug=self.workspace_slug,
            worker_name=self.worker_name,
            worker_id=self.worker_id,
        )


class WorkerStartedResponse(msgspec.Struct, frozen=True):
    worker_id: str


class ShardDescriptor(msgspec.Struct, frozen=True):
    shard_id: str = msgspec.field(name="shard_id")
    path: str
    start: int
    end: int

    @classmethod
    def from_shard(cls, shard: Shard) -> ShardDescriptor:
        return cls(shard_id=shard.id, path=shard.path, start=shard.start, end=shard.end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "path": self.path,
            "start": self.start,
            "end": self.end,
        }


class ShardClaimResponse(msgspec.Struct, frozen=True):
    shard: ShardDescriptor | None


class FinalizedShardWorker(msgspec.Struct, frozen=True):
    shard_id: str
    worker_id: str


class FinalizedShardWorkersResponse(msgspec.Struct, frozen=True):
    shards: list[FinalizedShardWorker]


class OkResponse(msgspec.Struct, frozen=True):
    ok: bool = True


@dataclass(frozen=True, slots=True)
class CloudRuntimeConfig:
    num_workers: int
    heartbeat_interval_seconds: int
    cpus_per_worker: int | None = None
    mem_mb_per_worker: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "num_workers": self.num_workers,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
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
class StagePayload:
    stage_index: int
    pipeline_payload: CloudPipelinePayload
    runtime: CloudRuntimeConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_index": self.stage_index,
            "pipeline_payload": self.pipeline_payload.to_dict(),
            "runtime": self.runtime.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CloudRunCreateRequest:
    name: str
    plan: dict[str, Any]
    stage_payloads: list[StagePayload]
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
            "stage_payloads": [payload.to_dict() for payload in self.stage_payloads],
        }
        if self.manifest is not None:
            payload["manifest"] = self.manifest
        return payload


class CloudRunCreateResponse(msgspec.Struct, frozen=True):
    job_id: str
    stage_index: int
    status: str
