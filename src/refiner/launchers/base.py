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
        return compile_planned_stages(
            self._resolved_stages(stages),
            secret_values=secret_values,
        )

    def _resolved_stages(
        self,
        stages: list[PlannedStage] | None = None,
    ) -> list[PlannedStage]:
        return [
            replace(stage, compute=self._stage_compute_requirements(stage.compute))
            for stage in (stages or self._planned_stages())
        ]

    def _stage_compute_requirements(
        self, compute: StageComputeRequirements
    ) -> StageComputeRequirements:
        if not compute.inherit_launcher_resources:
            return compute
        return replace(
            compute,
            cpus_per_worker=(
                compute.cpus_per_worker
                if compute.cpus_per_worker is not None
                else getattr(self, "cpus_per_worker", None)
            ),
            memory_mb_per_worker=(
                compute.memory_mb_per_worker
                if compute.memory_mb_per_worker is not None
                else getattr(self, "mem_mb_per_worker", None)
            ),
            gpus_per_worker=(
                compute.gpus_per_worker
                if compute.gpus_per_worker is not None
                else getattr(self, "gpus_per_worker", None)
            ),
            gpu_type=(
                compute.gpu_type
                if compute.gpu_type is not None
                else getattr(self, "gpu_type", None)
            ),
        )

    def _run_manifest(
        self, *, secret_values: tuple[str, ...] = ()
    ) -> dict[str, object]:
        return build_run_manifest(secret_values=secret_values)

    @abstractmethod
    def launch(self):
        raise NotImplementedError


__all__ = ["BaseLauncher"]
