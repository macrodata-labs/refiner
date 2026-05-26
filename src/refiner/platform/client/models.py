from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import msgspec

from refiner.pipeline.resources import GPU
from refiner.pipeline.data.shard import Shard
from refiner.services.base import RuntimeServiceSpec
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
    warnings: list[str] = msgspec.field(default_factory=list)

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
            warnings=[],
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


@dataclass(frozen=True, slots=True)
class CloudRuntimeConfig:
    num_workers: int
    cpus_per_worker: int | None = None
    mem_mb_per_worker: int | None = None
    gpu: GPU | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "num_workers": self.num_workers,
        }
        if self.cpus_per_worker is not None:
            payload["cpus_per_worker"] = self.cpus_per_worker
        if self.mem_mb_per_worker is not None:
            payload["mem_mb_per_worker"] = self.mem_mb_per_worker
        if self.gpu is not None:
            payload["gpu"] = self.gpu.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class CloudFile:
    """Reference to a workspace cloud file used in a cloud run payload."""

    file_id: str
    """Workspace-scoped cloud file UUID."""

    def to_dict(self) -> dict[str, Any]:
        return {"file_id": self.file_id}


@dataclass(frozen=True, slots=True)
class CloudFileUploadRequestItem:
    """Declared metadata for one file that needs an upload URL."""

    sha256: str
    """Lowercase hex SHA-256 digest of the file bytes."""

    size_bytes: int
    """Exact file size in bytes."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


class CloudFileUploadStatus(str, Enum):
    """Cloud file upload URL status returned by the public cloud API."""

    NEW = "new"
    """The file is not available yet and must be uploaded to the signed URL."""

    EXISTS = "exists"
    """The workspace already has the file; upload and completion can be skipped."""


class CloudFileUploadInstruction(msgspec.Struct, frozen=True):
    """Upload URL response item for one declared cloud file."""

    file_id: str
    """Workspace-scoped cloud file UUID assigned by the API."""

    sha256: str
    """Lowercase hex SHA-256 digest of the expected file bytes."""

    size_bytes: int
    """Exact expected file size in bytes."""

    status: CloudFileUploadStatus
    """Whether the file is new or already available in the workspace."""

    url: str | None = None
    """Opaque signed upload URL for new files."""

    required_headers: dict[str, str] | None = None
    """HTTP headers that must be sent exactly with the signed upload request."""

    upload_url_expires_at: datetime | None = None
    """Signed upload URL expiry time, when an upload URL exists."""

    expires_at: datetime | None = None
    """Cloud file expiry time, or None for non-expiring files."""

    def upload_target(self) -> tuple[str, dict[str, str]] | None:
        if self.status is CloudFileUploadStatus.EXISTS:
            return None
        if not self.url or self.required_headers is None:
            raise ValueError(
                "new cloud file upload instruction is missing upload target"
            )
        return self.url, self.required_headers


class CloudFileUploadUrlsResponse(msgspec.Struct, frozen=True):
    """Response from the cloud file upload URL endpoint."""

    files: list[CloudFileUploadInstruction]
    """Upload instructions corresponding to requested files."""


@dataclass(frozen=True, slots=True)
class CloudFileCompleteRequestItem:
    """Request item for marking one uploaded cloud file complete."""

    file_id: str
    """Workspace-scoped cloud file UUID to complete."""

    def to_dict(self) -> dict[str, Any]:
        return {"file_id": self.file_id}


class CloudFileCompleteResult(msgspec.Struct, frozen=True):
    """Completion result for one cloud file."""

    file_id: str
    """Workspace-scoped cloud file UUID."""

    sha256: str
    """Lowercase hex SHA-256 digest verified by the backend."""

    size_bytes: int
    """Verified file size in bytes."""

    uploaded_at: datetime
    """Time when the file was marked uploaded."""

    expires_at: datetime | None = None
    """Cloud file expiry time, or None for non-expiring files."""


class CloudFileCompleteResponse(msgspec.Struct, frozen=True):
    """Response from the cloud file completion endpoint."""

    files: list[CloudFileCompleteResult]
    """Completion results corresponding to requested files."""


@dataclass(frozen=True, slots=True)
class StagePayload:
    stage_index: int
    pipeline_payload: CloudFile
    runtime: CloudRuntimeConfig
    runtime_services: tuple[RuntimeServiceSpec, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage_index": self.stage_index,
            "pipeline_payload": self.pipeline_payload.to_dict(),
            "runtime": self.runtime.to_dict(),
        }
        if self.runtime_services:
            payload["runtime_services"] = [
                service.to_dict() for service in self.runtime_services
            ]
        return payload


@dataclass(frozen=True, slots=True)
class CloudRunCreateRequest:
    name: str
    plan: dict[str, Any]
    stage_payloads: list[StagePayload]
    manifest: dict[str, Any] | None = None
    sync_local_dependencies: bool = True
    secrets: list[dict[str, Any]] | None = None
    env: dict[str, str] | None = None
    continue_from_job: str | None = None
    unsafe_continue: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "executor": {
                "sync_local_dependencies": self.sync_local_dependencies,
            },
            "plan": self.plan,
            "stage_payloads": [payload.to_dict() for payload in self.stage_payloads],
        }
        if self.manifest is not None:
            payload["manifest"] = self.manifest
        if self.secrets:
            payload["secrets"] = self.secrets
        if self.env:
            payload["env"] = self.env
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
    warnings: list[str] = msgspec.field(default_factory=list)
