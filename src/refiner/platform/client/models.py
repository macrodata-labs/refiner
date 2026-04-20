from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import msgspec

from refiner.pipeline.data.shard import Shard
from refiner.worker.lifecycle import FinalizedShardWorker


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


class WorkerStartedResponse(msgspec.Struct, frozen=True):
    worker_id: str


class SerializedShard(msgspec.Struct, frozen=True):
    """Wire form of a shard: scheduling hints plus a serialized descriptor."""

    shard_id: str = msgspec.field(name="shard_id")
    descriptor: dict[str, Any]
    global_ordinal: int | None = None
    start_key: str | None = None
    end_key: str | None = None

    @classmethod
    def from_shard(cls, shard: Shard) -> SerializedShard:
        return cls(
            shard_id=shard.id,
            global_ordinal=shard.global_ordinal,
            start_key=shard.start_key,
            end_key=shard.end_key,
            descriptor=shard.descriptor.to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "global_ordinal": self.global_ordinal,
            "start_key": self.start_key,
            "end_key": self.end_key,
            "descriptor": self.descriptor,
        }


class ShardClaimResponse(msgspec.Struct, frozen=True):
    shard: SerializedShard | None


class FinalizedShardWorkersResponse(msgspec.Struct, frozen=True):
    shards: list[FinalizedShardWorker]


class OkResponse(msgspec.Struct, frozen=True):
    ok: bool = True


class StageLifecycleStage(msgspec.Struct, frozen=True):
    job_id: str
    index: int
    status: str


class StageLifecycleResponse(msgspec.Struct, frozen=True):
    stage: StageLifecycleStage


@dataclass(slots=True)
class CloudRuntimeConfig:
    num_workers: int
    cpus_per_worker: int | None = None
    mem_mb_per_worker: int | None = None
    gpus_per_worker: int | None = None
    gpu_type: str | None = None

    def __post_init__(self) -> None:
        if self.num_workers <= 0:
            raise ValueError("num_workers must be > 0")
        if self.cpus_per_worker is not None and self.cpus_per_worker <= 0:
            raise ValueError("cpus_per_worker must be > 0")
        if self.mem_mb_per_worker is not None and self.mem_mb_per_worker <= 0:
            raise ValueError("mem_mb_per_worker must be > 0")
        if self.gpus_per_worker is not None and self.gpus_per_worker <= 0:
            raise ValueError("gpus_per_worker must be > 0")
        normalized_gpu_type = (
            self.gpu_type.strip() if self.gpu_type is not None else None
        )
        if self.gpu_type is not None and not normalized_gpu_type:
            raise ValueError("gpu_type must be non-empty")
        if self.gpus_per_worker is not None and normalized_gpu_type is None:
            raise ValueError("gpu_type is required when gpus_per_worker is set")
        if normalized_gpu_type is not None and self.gpus_per_worker is None:
            raise ValueError("gpus_per_worker is required when gpu_type is set")
        self.gpu_type = normalized_gpu_type

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"num_workers": self.num_workers}
        if self.cpus_per_worker is not None:
            payload["cpus_per_worker"] = self.cpus_per_worker
        if self.mem_mb_per_worker is not None:
            payload["mem_mb_per_worker"] = self.mem_mb_per_worker
        if self.gpus_per_worker is not None:
            payload["gpus_per_worker"] = self.gpus_per_worker
        if self.gpu_type is not None:
            payload["gpu_type"] = self.gpu_type
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
    secrets: dict[str, str] | None = None
    continue_from_job: str | None = None
    unsafe_continue: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "executor": {
                "sync_local_dependencies": self.sync_local_dependencies,
            },
            "name": self.name,
            "plan": self.plan,
            "stage_payloads": [
                stage_payload.to_dict() for stage_payload in self.stage_payloads
            ],
        }
        if self.manifest is not None:
            payload["manifest"] = self.manifest
        if self.secrets:
            payload["secrets"] = self.secrets
        if self.continue_from_job is not None:
            payload["continue_from_job"] = self.continue_from_job
        if self.unsafe_continue:
            payload["unsafe_continue"] = True
        return payload


class CloudRunCreateResponse(msgspec.Struct, frozen=True):
    job_id: str
    stage_index: int
    status: str
    workspace_slug: str | None = msgspec.field(name="workspaceSlug", default=None)
