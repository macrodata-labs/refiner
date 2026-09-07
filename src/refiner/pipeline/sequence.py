from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from refiner.pipeline.resources import GPU
from refiner.pipeline.planning import WorkerCount

if TYPE_CHECKING:
    from refiner.launchers.cloud import CloudLaunchResult
    from refiner.launchers.local import LaunchStats
    from refiner.launchers.secrets import SecretInput
    from refiner.pipeline.pipeline import RefinerPipeline
    from refiner.pipeline.data.row import Row
    from refiner.pipeline.sinks.base import BaseSink
    from refiner.pipeline.sources.base import BaseSource
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
    inherit_launcher_resources: bool = False

    def __post_init__(self) -> None:
        normalized_name = _validate_stage_configuration(
            name=self.name,
            num_workers=self.num_workers,
            cpus_per_worker=self.cpus_per_worker,
            mem_mb_per_worker=self.mem_mb_per_worker,
        )
        object.__setattr__(self, "name", normalized_name)


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

    @property
    def primary_pipeline(self) -> RefinerPipeline:
        """Return the pipeline that produces the writer's primary output."""
        return self.stages[0].pipeline

    @property
    def source(self) -> BaseSource:
        return self.primary_pipeline.source

    @property
    def sink(self) -> BaseSink | None:
        return self.primary_pipeline.sink

    def iter_rows(self) -> Iterable[Row]:
        """Inspect rows from the primary pipeline without executing its writer."""
        return self.primary_pipeline.iter_rows()

    def __iter__(self) -> Iterator[Row]:
        return iter(self.iter_rows())

    def execute(
        self, rows: Iterable[Any], *, on_shard_delta: Any = None
    ) -> Iterable[Any]:
        return self.primary_pipeline.execute(rows, on_shard_delta=on_shard_delta)

    def output_schema(self) -> Any:
        return self.primary_pipeline.output_schema()

    def then(
        self,
        pipeline: RefinerPipeline | PipelineSequence,
        *,
        name: str,
        num_workers: WorkerCount = 1,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
        gpu: GPU | None = None,
    ) -> PipelineSequence:
        """Return a sequence with one named pipeline stage appended."""
        if isinstance(pipeline, PipelineSequence):
            appended = pipeline.as_stage(
                name=name,
                num_workers=num_workers,
                cpus_per_worker=cpus_per_worker,
                mem_mb_per_worker=mem_mb_per_worker,
                gpu=gpu,
            ).stages
        else:
            appended = (
                ConfiguredStage(
                    pipeline=pipeline,
                    name=name,
                    num_workers=num_workers,
                    cpus_per_worker=cpus_per_worker,
                    mem_mb_per_worker=mem_mb_per_worker,
                    gpu=gpu,
                ),
            )
        return PipelineSequence((*self.stages, *appended))

    def as_stage(
        self,
        *,
        name: str,
        num_workers: WorkerCount = 1,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
        gpu: GPU | None = None,
    ) -> PipelineSequence:
        """Rename and configure a writer-created sequence as one logical stage."""
        first, *remaining = self.stages
        renamed = [
            ConfiguredStage(
                pipeline=first.pipeline,
                name=name,
                num_workers=num_workers,
                cpus_per_worker=cpus_per_worker,
                mem_mb_per_worker=mem_mb_per_worker,
                gpu=gpu,
            )
        ]
        prefix = f"{first.name}_"
        for stage in remaining:
            suffix = stage.name.removeprefix(prefix)
            renamed.append(
                ConfiguredStage(
                    pipeline=stage.pipeline,
                    name=f"{name}_{suffix}",
                    num_workers=stage.num_workers,
                    cpus_per_worker=stage.cpus_per_worker,
                    mem_mb_per_worker=stage.mem_mb_per_worker,
                    gpu=stage.gpu,
                    inherit_launcher_resources=stage.inherit_launcher_resources,
                )
            )
        return PipelineSequence(renamed)

    def launch_local(
        self,
        *,
        name: str,
        num_workers: WorkerCount = 1,
        rundir: str | None = None,
        gpu: GPU | None = None,
    ) -> LaunchStats:
        """Run the configured stages sequentially on the local machine."""
        from refiner.launchers.local import LocalLauncher

        return LocalLauncher(
            pipeline=self,
            name=name,
            num_workers=num_workers,
            rundir=rundir,
            gpu=gpu,
        ).launch()

    def launch_cloud(
        self,
        *,
        name: str,
        provider: str = "modal",
        num_workers: WorkerCount = 1,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
        gpu: GPU | None = None,
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
            num_workers=num_workers,
            cpus_per_worker=cpus_per_worker,
            mem_mb_per_worker=mem_mb_per_worker,
            gpu=gpu,
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


__all__ = ["ConfiguredStage", "PipelineSequence"]
