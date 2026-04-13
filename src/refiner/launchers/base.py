from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING
from uuid import uuid4

from refiner.pipeline.planning import (
    PlannedStage,
    StageComputeRequirements,
    compile_planned_stages,
    plan_pipeline_stages,
)
from refiner.platform.client.api import MacrodataClient, sanitize_terminal_text
from refiner.platform.manifest import build_run_manifest

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline


class BaseLauncher(ABC):
    def __init__(
        self,
        *,
        pipeline: RefinerPipeline,
        name: str,
        num_workers: int = 1,
        cpus_per_worker: int | None = None,
        gpus_per_worker: int | None = None,
    ):
        if not name.strip():
            raise ValueError("name must be non-empty")
        self.pipeline = pipeline
        self.name = name
        if num_workers <= 0:
            raise ValueError("num_workers must be > 0")
        self.num_workers = num_workers
        if cpus_per_worker is not None and cpus_per_worker <= 0:
            raise ValueError("cpus_per_worker must be > 0")
        self.cpus_per_worker = cpus_per_worker
        if gpus_per_worker is not None and gpus_per_worker <= 0:
            raise ValueError("gpus_per_worker must be > 0")
        self.gpus_per_worker = gpus_per_worker

    @staticmethod
    def _build_local_job_id(name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-") or "job"
        return f"{slug}-{int(time.time())}-{uuid4().hex[:8]}"

    def _job_tracking_url(
        self, *, client: MacrodataClient, job_id: str, workspace_slug: str | None = None
    ) -> str:
        safe_base_url = sanitize_terminal_text(client.base_url).strip().rstrip("/")
        safe_job_id = sanitize_terminal_text(job_id).strip() or job_id
        safe_workspace_slug = (
            sanitize_terminal_text(workspace_slug).strip() if workspace_slug else None
        )
        if safe_workspace_slug:
            return f"{safe_base_url}/jobs/{safe_workspace_slug}/{safe_job_id}"
        return f"{safe_base_url}/jobs/{safe_job_id}"

    def _planned_stages(self) -> list[PlannedStage]:
        requested_workers = getattr(self, "num_workers", None)
        default_num_workers = (
            requested_workers if isinstance(requested_workers, int) else 1
        )
        return plan_pipeline_stages(
            self.pipeline,
            default_num_workers=default_num_workers,
        )

    def _compiled_plan(
        self,
        stages: list[PlannedStage] | None = None,
        *,
        secret_values: tuple[str, ...] = (),
    ) -> dict[str, object]:
        resolved_stages = [
            replace(stage, compute=self._stage_compute_requirements(stage.compute))
            for stage in (stages or self._planned_stages())
        ]
        return compile_planned_stages(resolved_stages, secret_values=secret_values)

    def _stage_compute_requirements(
        self, compute: StageComputeRequirements
    ) -> StageComputeRequirements:
        return replace(
            compute,
            cpus_per_worker=getattr(self, "cpus_per_worker", None),
            memory_mb_per_worker=getattr(self, "mem_mb_per_worker", None),
            gpus_per_worker=getattr(self, "gpus_per_worker", None),
            gpu_type=getattr(self, "gpu_type", None),
        )

    def _run_manifest(
        self, *, secret_values: tuple[str, ...] = ()
    ) -> dict[str, object]:
        return build_run_manifest(secret_values=secret_values)

    @abstractmethod
    def launch(self):
        raise NotImplementedError


__all__ = ["BaseLauncher"]
