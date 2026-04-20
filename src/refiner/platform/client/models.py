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


def _validate_cloud_runtime_fields(
    *,
    num_workers: int | None = None,
    cpus_per_worker: int | None = None,
    mem_mb_per_worker: int | None = None,
    gpus_per_worker: int | None = None,
    gpu_type: str | None = None,
    require_complete_gpu: bool = True,
) -> str | None:
    if num_workers is not None and num_workers <= 0:
        raise ValueError("num_workers must be > 0")
    if cpus_per_worker is not None and cpus_per_worker <= 0:
        raise ValueError("cpus_per_worker must be > 0")
    if mem_mb_per_worker is not None and mem_mb_per_worker <= 0:
        raise ValueError("mem_mb_per_worker must be > 0")
    if gpus_per_worker is not None and gpus_per_worker <= 0:
        raise ValueError("gpus_per_worker must be > 0")
    normalized_gpu_type = gpu_type.strip() if gpu_type is not None else None
    if gpu_type is not None and not normalized_gpu_type:
        raise ValueError("gpu_type must be non-empty")
    if (
        require_complete_gpu
        and gpus_per_worker is not None
        and normalized_gpu_type is None
    ):
        raise ValueError("gpu_type is required when gpus_per_worker is set")
    if (
        require_complete_gpu
        and normalized_gpu_type is not None
        and gpus_per_worker is None
    ):
        raise ValueError("gpus_per_worker is required when gpu_type is set")
    return normalized_gpu_type


def _cloud_runtime_payload(
    *,
    num_workers: int | None = None,
    cpus_per_worker: int | None = None,
    mem_mb_per_worker: int | None = None,
    gpus_per_worker: int | None = None,
    gpu_type: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if num_workers is not None:
        payload["num_workers"] = num_workers
    if cpus_per_worker is not None:
        payload["cpus_per_worker"] = cpus_per_worker
    if mem_mb_per_worker is not None:
        payload["mem_mb_per_worker"] = mem_mb_per_worker
    if gpus_per_worker is not None:
        payload["gpus_per_worker"] = gpus_per_worker
    if gpu_type is not None:
        payload["gpu_type"] = gpu_type
    return payload


def _update_cloud_run_request_payload(
    payload: dict[str, Any],
    *,
    executor: dict[str, Any] | None = None,
    name: str | None = None,
    plan: dict[str, Any] | None = None,
    stage_payloads: list[Any] | None = None,
    manifest: dict[str, Any] | None = None,
    secrets: dict[str, str] | None = None,
) -> dict[str, Any]:
    if executor is not None:
        payload["executor"] = executor
    if name is not None:
        payload["name"] = name
    if plan is not None:
        payload["plan"] = plan
    if stage_payloads is not None:
        payload["stage_payloads"] = [
            stage_payload.to_dict() for stage_payload in stage_payloads
        ]
    if manifest is not None:
        payload["manifest"] = manifest
    if secrets:
        payload["secrets"] = secrets
    return payload


class _CloudRuntimePayloadMixin:
    num_workers: int | None
    cpus_per_worker: int | None
    mem_mb_per_worker: int | None
    gpus_per_worker: int | None
    gpu_type: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gpu_type",
            _validate_cloud_runtime_fields(
                num_workers=self.num_workers,
                cpus_per_worker=self.cpus_per_worker,
                mem_mb_per_worker=self.mem_mb_per_worker,
                gpus_per_worker=self.gpus_per_worker,
                gpu_type=self.gpu_type,
                require_complete_gpu=False,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return _cloud_runtime_payload(
            num_workers=self.num_workers,
            cpus_per_worker=self.cpus_per_worker,
            mem_mb_per_worker=self.mem_mb_per_worker,
            gpus_per_worker=self.gpus_per_worker,
            gpu_type=self.gpu_type,
        )


@dataclass(frozen=True, slots=True)
class CloudRuntimeConfig(_CloudRuntimePayloadMixin):
    num_workers: int
    cpus_per_worker: int | None = None
    mem_mb_per_worker: int | None = None
    gpus_per_worker: int | None = None
    gpu_type: str | None = None


@dataclass(frozen=True, slots=True)
class CloudRuntimeOverrides(_CloudRuntimePayloadMixin):
    num_workers: int | None = None
    cpus_per_worker: int | None = None
    mem_mb_per_worker: int | None = None
    gpus_per_worker: int | None = None
    gpu_type: str | None = None


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
    runtime: CloudRuntimeConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage_index": self.stage_index,
            "pipeline_payload": self.pipeline_payload.to_dict(),
        }
        if self.runtime is not None:
            payload["runtime"] = self.runtime.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class CloudRunCreateRequest:
    name: str
    plan: dict[str, Any]
    stage_payloads: list[StagePayload]
    manifest: dict[str, Any] | None = None
    sync_local_dependencies: bool = True
    secrets: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _update_cloud_run_request_payload(
            {},
            executor={
                "type": "macrodata-cloud",
                "sync_local_dependencies": self.sync_local_dependencies,
            },
            name=self.name,
            plan=self.plan,
            stage_payloads=self.stage_payloads,
            manifest=self.manifest,
            secrets=self.secrets,
        )


@dataclass(frozen=True, slots=True)
class CloudResumeSelector:
    job_id: str | None = None
    latest_compatible: bool = False
    name: str | None = None
    limit_to_me: bool = False

    def __post_init__(self) -> None:
        has_job_id = self.job_id is not None
        if has_job_id == self.latest_compatible:
            raise ValueError(
                "resume selector must specify exactly one of job_id or latest_compatible"
            )
        if has_job_id:
            normalized_job_id = self.job_id.strip() if self.job_id is not None else None
            if not normalized_job_id:
                raise ValueError("job_id must be non-empty")
            if self.name is not None or self.limit_to_me:
                raise ValueError(
                    "name and limit_to_me are only supported with latest_compatible"
                )
            object.__setattr__(self, "job_id", normalized_job_id)
        if self.name is not None:
            normalized_name = self.name.strip()
            if not normalized_name:
                raise ValueError("resume selector name must be non-empty")
            object.__setattr__(self, "name", normalized_name)

    def to_dict(self) -> dict[str, Any]:
        if self.job_id is not None:
            return {"job_id": self.job_id}
        payload: dict[str, Any] = {"latest_compatible": True}
        if self.name is not None:
            payload["name"] = self.name
        if self.limit_to_me:
            payload["limit_to_me"] = True
        return payload


def build_resume_selector(
    *,
    job_id: str | None = None,
    latest_compatible: bool = False,
    name: str | None = None,
    limit_to_me: bool = False,
) -> CloudResumeSelector | None:
    if job_id is None and not latest_compatible:
        if name is not None or limit_to_me:
            raise ValueError(
                "resume_name and resume_limit_to_me require resume='latest-compatible'"
            )
        return None
    return CloudResumeSelector(
        job_id=job_id,
        latest_compatible=latest_compatible,
        name=name,
        limit_to_me=limit_to_me,
    )


@dataclass(frozen=True, slots=True)
class CloudRunResumeRequest:
    selector: CloudResumeSelector
    plan: dict[str, Any]
    stage_payloads: list[StagePayload]
    manifest: dict[str, Any]
    sync_local_dependencies: bool
    name: str | None = None
    runtime_overrides: CloudRuntimeOverrides | None = None
    secrets: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.name is not None:
            normalized_name = self.name.strip()
            if not normalized_name:
                raise ValueError("name must be non-empty")
            object.__setattr__(self, "name", normalized_name)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = _update_cloud_run_request_payload(
            {"selector": self.selector.to_dict()},
            executor={
                "type": "macrodata-cloud",
                "sync_local_dependencies": self.sync_local_dependencies,
            },
            name=self.name,
            plan=self.plan,
            stage_payloads=self.stage_payloads,
            manifest=self.manifest,
            secrets=self.secrets,
        )
        if self.runtime_overrides is not None:
            payload["runtime_overrides"] = self.runtime_overrides.to_dict()
        return payload


class CloudRunCreateResponse(msgspec.Struct, frozen=True):
    job_id: str
    stage_index: int
    status: str
    workspace_slug: str | None = msgspec.field(name="workspaceSlug", default=None)
