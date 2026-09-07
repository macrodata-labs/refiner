from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from refiner.pipeline.resources import GPU
from refiner.pipeline.planning import WorkerCount

if TYPE_CHECKING:
    from refiner.launchers.cloud import CloudLaunchResult
    from refiner.launchers.local import LaunchStats
    from refiner.launchers.secrets import SecretInput
    from refiner.pipeline.pipeline import RefinerPipeline
    from refiner.pipeline.sinks.base import BaseSink
    from refiner.platform.client import CloudProvider, CloudRegion


def _validate_stage_configuration(
    *,
    name: str,
    num_workers: WorkerCount,
    cpus_per_worker: int | None,
    mem_mb_per_worker: int | None,
) -> str:
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("stage name must be non-empty")
    if num_workers != "auto" and num_workers <= 0:
        raise ValueError("num_workers must be > 0 or 'auto'")
    if cpus_per_worker is not None and cpus_per_worker <= 0:
        raise ValueError("cpus_per_worker must be > 0")
    if mem_mb_per_worker is not None and mem_mb_per_worker <= 0:
        raise ValueError("mem_mb_per_worker must be > 0")
    return normalized_name


@dataclass(frozen=True, slots=True)
class ConfiguredStage:
    """A named pipeline and the resources assigned to its execution stage."""

    pipeline: RefinerPipeline
    name: str
    num_workers: WorkerCount
    cpus_per_worker: int | None
    mem_mb_per_worker: int | None
    gpu: GPU | None

    def __post_init__(self) -> None:
        normalized_name = _validate_stage_configuration(
            name=self.name,
            num_workers=self.num_workers,
            cpus_per_worker=self.cpus_per_worker,
            mem_mb_per_worker=self.mem_mb_per_worker,
        )
        object.__setattr__(self, "name", normalized_name)


@dataclass(frozen=True, slots=True)
class FollowupStage:
    """A writer-owned stage that runs after its parent stage succeeds."""

    pipeline: RefinerPipeline
    name: str
    num_workers: int = 1
    cpus_per_worker: int | None = None
    mem_mb_per_worker: int | None = None
    gpu: GPU | None = None

    def __post_init__(self) -> None:
        normalized_name = _validate_stage_configuration(
            name=self.name,
            num_workers=self.num_workers,
            cpus_per_worker=self.cpus_per_worker,
            mem_mb_per_worker=self.mem_mb_per_worker,
        )
        object.__setattr__(self, "name", normalized_name)

    @classmethod
    def from_sink(
        cls,
        *,
        name: str,
        sink: BaseSink,
        num_workers: int = 1,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
        gpu: GPU | None = None,
    ) -> FollowupStage:
        """Create a task-backed follow-up stage for a finalizer sink."""
        from refiner.pipeline.pipeline import RefinerPipeline
        from refiner.pipeline.sources.task import TaskSource

        return cls(
            pipeline=RefinerPipeline(
                source=TaskSource(num_tasks=num_workers), sink=sink
            ),
            name=name,
            num_workers=num_workers,
            cpus_per_worker=cpus_per_worker,
            mem_mb_per_worker=mem_mb_per_worker,
            gpu=gpu,
        )


@dataclass(frozen=True, slots=True, init=False)
class PipelineSequence:
    """An immutable, ordered collection of independently executable pipelines."""

    stages: tuple[ConfiguredStage, ...]

    def __init__(self, stages: Sequence[ConfiguredStage]) -> None:
        normalized_stages = tuple(stages)
        if not normalized_stages:
            raise ValueError("a pipeline sequence must contain at least one stage")
        names = [stage.name for stage in normalized_stages]
        if len(set(names)) != len(names):
            raise ValueError("stage names must be unique")
        object.__setattr__(self, "stages", normalized_stages)

    def then(
        self,
        pipeline: RefinerPipeline,
        *,
        name: str,
        num_workers: WorkerCount = 1,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
        gpu: GPU | None = None,
    ) -> PipelineSequence:
        """Return a sequence with one named pipeline stage appended."""
        return PipelineSequence(
            (
                *self.stages,
                ConfiguredStage(
                    pipeline=pipeline,
                    name=name,
                    num_workers=num_workers,
                    cpus_per_worker=cpus_per_worker,
                    mem_mb_per_worker=mem_mb_per_worker,
                    gpu=gpu,
                ),
            )
        )

    def launch_local(
        self,
        *,
        name: str,
        rundir: str | None = None,
    ) -> LaunchStats:
        """Run the configured stages sequentially on the local machine."""
        from refiner.launchers.local import LocalLauncher

        return LocalLauncher(
            pipeline=self,
            name=name,
            rundir=rundir,
        ).launch()

    def launch_cloud(
        self,
        *,
        name: str,
        provider: str = "modal",
        cloud: CloudProvider = "aws",
        region: CloudRegion | Sequence[CloudRegion] = ("us", "eu", "ca"),
        sync_local_dependencies: bool = False,
        dependencies: Sequence[str] | None = None,
        refiner_extras: Sequence[str] | None = None,
        secrets: SecretInput | None = None,
        env: Mapping[str, object | None] | None = None,
        continue_from_job: str | None = None,
        unsafe_continue: bool = False,
    ) -> CloudLaunchResult:
        """Launch the configured stages sequentially on Macrodata Cloud."""
        from refiner.launchers.cloud import CloudLauncher

        return CloudLauncher(
            pipeline=self,
            name=name,
            provider=provider,
            cloud=cloud,
            region=region,
            sync_local_dependencies=sync_local_dependencies,
            dependencies=dependencies,
            refiner_extras=refiner_extras,
            secrets=secrets,
            env=dict(env) if env is not None else None,
            continue_from_job=continue_from_job,
            unsafe_continue=unsafe_continue,
        ).launch()


__all__ = ["ConfiguredStage", "FollowupStage", "PipelineSequence"]
