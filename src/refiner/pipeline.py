from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, List, Literal

from fsspec import AbstractFileSystem

from refiner.io.fileset import DataFileSetLike
from refiner.processors.step import (
    BatchFn,
    BatchStep,
    FlatMapFn,
    FlatMapStep,
    FnBatchStep,
    FnFlatMapStep,
    FnRowStep,
    MapFn,
    RefinerStep,
    RowStep,
    normalize_batch_item,
    normalize_row_result,
)
from refiner.readers import CsvReader, JsonlReader, ParquetReader
from refiner.readers.base import BaseReader
from refiner.readers.row import Row
from refiner.runtime.row_queue import RowQueue
from refiner.readers.utils import DEFAULT_TARGET_SHARD_BYTES

if TYPE_CHECKING:
    from refiner.runtime.launchers.cloud import CloudLaunchResult
    from refiner.runtime.launchers.local import LaunchStats


class RefinerPipeline:
    source: BaseReader
    pipeline_steps: List[RefinerStep]

    def __init__(
        self, source: BaseReader, pipeline_steps: List[RefinerStep] | None = None
    ):
        self.source = source
        self.pipeline_steps = list(pipeline_steps) if pipeline_steps else []

    def add_step(self, step: RefinerStep) -> "RefinerPipeline":
        return self.__class__(self.source, self.pipeline_steps + [step])

    def map(self, fn: MapFn) -> "RefinerPipeline":
        return self.add_step(FnRowStep(fn=fn, op_name="map"))

    def batch_map(self, fn: BatchFn, *, batch_size: int) -> "RefinerPipeline":
        if batch_size <= 1:
            raise ValueError("batch_size for batch_map must be > 1")
        return self.add_step(
            FnBatchStep(fn=fn, batch_size=batch_size, op_name="batch_map")
        )

    def flat_map(self, fn: FlatMapFn) -> "RefinerPipeline":
        return self.add_step(FnFlatMapStep(fn=fn, op_name="flat_map"))

    def filter(self, predicate: Callable[[Row], bool]) -> "RefinerPipeline":
        return self.add_step(
            FnFlatMapStep(
                fn=lambda row: [row] if predicate(row) else [],
                op_name="filter",
            )
        )

    def execute_rows(self, rows: Iterable[Row]) -> Iterable[Row]:
        """Execute rows with per-step queues and step-local batch triggering."""
        steps = tuple(self.pipeline_steps)
        if not steps:
            for row in rows:
                yield row
            return

        queues: list[RowQueue] = [RowQueue() for _ in range(len(steps) + 1)]
        scratch: list[list[Row]] = [[] for _ in steps]

        def _run_step(i: int, *, flush_all: bool) -> None:
            step = steps[i]
            inp = queues[i]
            if not inp:
                return
            out = queues[i + 1]

            if isinstance(step, RowStep):
                for row in inp.take_all():
                    normalized = normalize_row_result(row, step.apply_row(row))
                    out.append(normalized)
                return

            if isinstance(step, FlatMapStep):
                tmp = scratch[i]
                tmp.clear()
                for row in inp.take_all():
                    for item in step.apply_row_many(row):
                        normalized = normalize_batch_item(item)
                        if normalized is not None:
                            tmp.append(normalized)
                out.extend(tmp)
                return

            if isinstance(step, BatchStep):
                if flush_all:
                    batch_in = inp.take_all()
                else:
                    n = (len(inp) // step.batch_size) * step.batch_size
                    if n == 0:
                        return
                    batch_in = inp.take(n)
                if not batch_in:
                    return
                tmp = scratch[i]
                tmp.clear()
                for item in step.apply_batch(batch_in):
                    normalized = normalize_batch_item(item)
                    if normalized is not None:
                        tmp.append(normalized)
                out.extend(tmp)
                return

            raise TypeError(f"Unsupported step type: {type(step)!r}")

        def _pump(flush_all: bool) -> None:
            for i in range(len(steps)):
                _run_step(i, flush_all=flush_all)

        def _drain_output() -> Iterable[Row]:
            outq = queues[-1]
            if not outq:
                return
            for row in outq.take_all():
                yield row

        for row in rows:
            queues[0].append(row)
            _pump(flush_all=False)
            for out in _drain_output():
                yield out

        _pump(flush_all=True)
        for out in _drain_output():
            yield out

    def iter_rows(self) -> Iterable[Row]:
        """Local execution mode: lazily process all shards and yield output rows."""
        return self.execute_rows(self.source.read())

    def materialize(self) -> list[Row]:
        """Compute all output rows into memory (local/dev utility)."""
        return list(self.iter_rows())

    def take(self, n: int) -> list[Row]:
        """Return up to the first `n` rows from local execution."""
        if n < 0:
            raise ValueError("n must be >= 0")
        out: list[Row] = []
        for row in self.iter_rows():
            out.append(row)
            if len(out) >= n:
                break
        return out

    def __iter__(self):
        return iter(self.iter_rows())

    def launch_local(
        self,
        *,
        name: str,
        num_workers: int = 1,
        workdir: str | None = None,
        heartbeat_every_rows: int = 4096,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
    ) -> "LaunchStats":
        """Launch the pipeline locally.

        Args:
            name: Human-readable run name.
            num_workers: Number of local worker processes.
            workdir: Optional working directory for ledger and run artifacts.
            heartbeat_every_rows: Heartbeat cadence for worker progress reporting.
            cpus_per_worker: Optional CPU cores pinned per worker.
            mem_mb_per_worker: Optional per-worker soft memory limit in MB.
        """
        from refiner.runtime.launchers.local import LocalLauncher

        launcher = LocalLauncher(
            pipeline=self,
            name=name,
            num_workers=num_workers,
            workdir=workdir,
            heartbeat_every_rows=heartbeat_every_rows,
            cpus_per_worker=cpus_per_worker,
            mem_mb_per_worker=mem_mb_per_worker,
        )
        return launcher.launch()

    def launch_cloud(
        self,
        *,
        name: str,
        num_workers: int = 1,
        heartbeat_every_rows: int = 4096,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
    ) -> "CloudLaunchResult":
        """Launch the pipeline on Macrodata Cloud.

        Args:
            name: Human-readable run name.
            num_workers: Requested logical worker count.
            heartbeat_every_rows: Worker heartbeat cadence.
            cpus_per_worker: Optional requested CPU cores per worker.
            mem_mb_per_worker: Optional requested memory per worker in MB.
        """
        from refiner.runtime.launchers.cloud import CloudLauncher

        launcher = CloudLauncher(
            pipeline=self,
            name=name,
            num_workers=num_workers,
            heartbeat_every_rows=heartbeat_every_rows,
            cpus_per_worker=cpus_per_worker,
            mem_mb_per_worker=mem_mb_per_worker,
        )
        return launcher.launch()


## readers
def read_csv(
    inputs: DataFileSetLike,
    *,
    fs: AbstractFileSystem | None = None,
    storage_options: Mapping[str, Any] | None = None,
    recursive: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    multiline_rows: bool = False,
    encoding: str = "utf-8",
    sharding_mode: Literal["bytes_lazy", "scan"] = "bytes_lazy",
) -> RefinerPipeline:
    """Create a pipeline with a CSV reader source."""
    return RefinerPipeline(
        source=CsvReader(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            target_shard_bytes=target_shard_bytes,
            multiline_rows=multiline_rows,
            encoding=encoding,
            sharding_mode=sharding_mode,
        )
    )


def read_jsonl(
    inputs: DataFileSetLike,
    *,
    fs: AbstractFileSystem | None = None,
    storage_options: Mapping[str, Any] | None = None,
    recursive: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
) -> RefinerPipeline:
    """Create a pipeline with a JSONL reader source."""
    return RefinerPipeline(
        source=JsonlReader(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            target_shard_bytes=target_shard_bytes,
        )
    )


def read_parquet(
    inputs: DataFileSetLike,
    *,
    fs: AbstractFileSystem | None = None,
    storage_options: Mapping[str, Any] | None = None,
    recursive: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    arrow_batch_size: int = 65536,
    columns_to_read: Sequence[str] | None = None,
    sharding_mode: Literal["rowgroups", "bytes_lazy"] = "rowgroups",
) -> RefinerPipeline:
    """Create a pipeline with a Parquet reader source."""
    return RefinerPipeline(
        source=ParquetReader(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            target_shard_bytes=target_shard_bytes,
            arrow_batch_size=arrow_batch_size,
            columns_to_read=columns_to_read,
            sharding_mode=sharding_mode,
        )
    )
