from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal

from fsspec import AbstractFileSystem

from refiner.expressions import Expr, col as col_expr, lit
from refiner.io.datafolder import DataFolderLike
from refiner.io.fileset import DataFileSetLike
from refiner.processors.step import (
    AsyncMapFn,
    BatchFn,
    CastStep,
    DropStep,
    FilterExprStep,
    FilterRowStep,
    FlatMapFn,
    FnAsyncRowStep,
    FnBatchStep,
    FnFlatMapStep,
    FnRowStep,
    MapFn,
    RenameStep,
    RefinerStep,
    SelectStep,
    VectorizedOp,
    VectorizedSegmentStep,
    WithColumnsStep,
)
from refiner.sources import (
    BaseSource,
    CsvReader,
    ItemsSource,
    JsonlReader,
    LeRobotEpisodeReader,
    ParquetReader,
    TaskSource,
)
from refiner.sources.row import Row
from refiner.runtime.execution.engine import (
    Block,
    Segment,
    compile_segments,
    execute_segments,
    iter_rows,
)
from refiner.runtime.execution.row_steps import ShardDeltaFn
from refiner.runtime.types import SourceUnit
from refiner.sources.readers.utils import DEFAULT_TARGET_SHARD_BYTES

if TYPE_CHECKING:
    from refiner.runtime.launchers.cloud import CloudLaunchResult
    from refiner.runtime.launchers.local import LaunchStats
    from refiner.runtime.sinks import BaseSink


class RefinerPipeline:
    source: BaseSource
    pipeline_steps: tuple[RefinerStep, ...]
    _compiled_segments: tuple[Segment, ...] | None
    max_vectorized_block_bytes: int | None
    sink: "BaseSink | None"

    def __init__(
        self,
        source: BaseSource,
        pipeline_steps: Sequence[RefinerStep] | None = None,
        *,
        max_vectorized_block_bytes: int | None = None,
        sink: "BaseSink | None" = None,
    ):
        if max_vectorized_block_bytes is not None and max_vectorized_block_bytes <= 0:
            raise ValueError("max_vectorized_block_bytes must be > 0 when provided")
        self.source = source
        self.pipeline_steps = tuple(pipeline_steps) if pipeline_steps else ()
        self._compiled_segments = None
        self.max_vectorized_block_bytes = max_vectorized_block_bytes
        self.sink = sink

    def add_step(self, step: RefinerStep) -> "RefinerPipeline":
        return self.__class__(
            self.source,
            self.pipeline_steps + (step,),
            max_vectorized_block_bytes=self.max_vectorized_block_bytes,
            sink=self.sink,
        )

    def _add_vectorized_op(self, op: VectorizedOp) -> "RefinerPipeline":
        # Fuse adjacent expression-backed operations so each fused segment does
        # one row->Arrow and Arrow->row conversion boundary.
        if self.pipeline_steps and isinstance(
            self.pipeline_steps[-1], VectorizedSegmentStep
        ):
            prev = self.pipeline_steps[-1]
            merged = VectorizedSegmentStep(ops=prev.ops + (op,))
            return self.__class__(
                self.source,
                self.pipeline_steps[:-1] + (merged,),
                max_vectorized_block_bytes=self.max_vectorized_block_bytes,
                sink=self.sink,
            )
        return self.add_step(VectorizedSegmentStep(ops=(op,)))

    def with_max_vectorized_block_bytes(
        self, max_vectorized_block_bytes: int | None
    ) -> "RefinerPipeline":
        return self.__class__(
            self.source,
            self.pipeline_steps,
            max_vectorized_block_bytes=max_vectorized_block_bytes,
            sink=self.sink,
        )

    def with_sink(self, sink: "BaseSink | None") -> "RefinerPipeline":
        return self.__class__(
            self.source,
            self.pipeline_steps,
            max_vectorized_block_bytes=self.max_vectorized_block_bytes,
            sink=sink,
        )

    def write_jsonl(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}.jsonl",
    ) -> "RefinerPipeline":
        from refiner.runtime.sinks import JsonlSink

        return self.with_sink(
            JsonlSink(output=output, filename_template=filename_template)
        )

    def write_parquet(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}.parquet",
        compression: str | None = None,
    ) -> "RefinerPipeline":
        from refiner.runtime.sinks import ParquetSink

        return self.with_sink(
            ParquetSink(
                output=output,
                filename_template=filename_template,
                compression=compression,
            )
        )

    def _get_compiled_segments(self) -> tuple[Segment, ...]:
        if self._compiled_segments is None:
            self._compiled_segments = compile_segments(self.pipeline_steps)
        return self._compiled_segments

    def map(self, fn: MapFn) -> "RefinerPipeline":
        return self.add_step(
            FnRowStep(fn=fn, op_name="map", index=len(self.pipeline_steps) + 1)
        )

    def map_async(
        self,
        fn: AsyncMapFn,
        *,
        max_in_flight: int = 16,
        preserve_order: bool = True,
    ) -> "RefinerPipeline":
        return self.add_step(
            FnAsyncRowStep(
                fn=fn,
                max_in_flight=max_in_flight,
                preserve_order=preserve_order,
                op_name="map_async",
                index=len(self.pipeline_steps) + 1,
            )
        )

    def batch_map(self, fn: BatchFn, *, batch_size: int) -> "RefinerPipeline":
        if batch_size <= 1:
            raise ValueError("batch_size for batch_map must be > 1")
        return self.add_step(
            FnBatchStep(
                fn=fn,
                batch_size=batch_size,
                op_name="batch_map",
                index=len(self.pipeline_steps) + 1,
            )
        )

    def flat_map(self, fn: FlatMapFn) -> "RefinerPipeline":
        return self.add_step(
            FnFlatMapStep(fn=fn, op_name="flat_map", index=len(self.pipeline_steps) + 1)
        )

    def filter(self, predicate: Callable[[Row], bool] | Expr) -> "RefinerPipeline":
        if isinstance(predicate, Expr):
            return self._add_vectorized_op(FilterExprStep(predicate=predicate))
        return self.add_step(
            FilterRowStep(
                predicate=predicate,
                op_name="filter",
                index=len(self.pipeline_steps) + 1,
            )
        )

    def select(self, *columns: str) -> "RefinerPipeline":
        if not columns:
            raise ValueError("select requires at least one column")
        return self._add_vectorized_op(SelectStep(columns=tuple(columns)))

    def with_columns(self, **assignments: Expr | Any) -> "RefinerPipeline":
        if not assignments:
            raise ValueError("with_columns requires at least one assignment")
        exprs = {
            name: value if isinstance(value, Expr) else lit(value)
            for name, value in assignments.items()
        }
        return self._add_vectorized_op(WithColumnsStep(assignments=exprs))

    def with_column(self, name: str, value: Expr | Any) -> "RefinerPipeline":
        expr = value if isinstance(value, Expr) else lit(value)
        return self._add_vectorized_op(WithColumnsStep(assignments={name: expr}))

    def drop(self, *columns: str) -> "RefinerPipeline":
        if not columns:
            raise ValueError("drop requires at least one column")
        return self._add_vectorized_op(DropStep(columns=tuple(columns)))

    def rename(self, **mapping: str) -> "RefinerPipeline":
        if not mapping:
            raise ValueError("rename requires at least one mapping")
        return self._add_vectorized_op(RenameStep(mapping=mapping))

    def cast(self, **dtypes: str) -> "RefinerPipeline":
        if not dtypes:
            raise ValueError("cast requires at least one dtype mapping")
        return self._add_vectorized_op(CastStep(dtypes=dtypes))

    def execute(
        self,
        rows: Iterable[SourceUnit],
        *,
        on_shard_delta: ShardDeltaFn | None = None,
    ) -> Iterable[Block]:
        """Execute source stream through compiled segments.

        Returns internal execution blocks (row blocks or Arrow blocks).
        Use `iter_rows()` to force row iteration.

        Note:
            This method is computation-only and does not write through `self.sink`.
            Sink writes happen in worker runtime/launch paths.
        """
        yield from execute_segments(
            rows,
            self._get_compiled_segments(),
            max_vectorized_block_bytes=self.max_vectorized_block_bytes,
            on_shard_delta=on_shard_delta,
        )

    def iter_rows(self) -> Iterable[Row]:
        """Local execution mode: lazily process all shards and yield output rows.

        This method ignores `self.sink` by design.
        """
        return iter_rows(self.execute(self.source.read()))

    def materialize(self) -> list[Row]:
        """Compute all output rows into memory (local/dev utility).

        This method ignores `self.sink` by design.
        """
        return list(self.iter_rows())

    def take(self, n: int) -> list[Row]:
        """Return up to the first `n` rows from local execution.

        This method ignores `self.sink` by design.
        """
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
        sync_local_dependencies: bool = True,
    ) -> "CloudLaunchResult":
        """Launch the pipeline on Macrodata Cloud.

        Args:
            name: Human-readable run name.
            num_workers: Requested logical worker count.
            heartbeat_every_rows: Worker heartbeat cadence.
            cpus_per_worker: Optional requested CPU cores per worker.
            mem_mb_per_worker: Optional requested memory per worker in MB.
            sync_local_dependencies: Sync submitting environment dependencies in cloud image.
        """
        from refiner.runtime.launchers.cloud import CloudLauncher

        launcher = CloudLauncher(
            pipeline=self,
            name=name,
            num_workers=num_workers,
            heartbeat_every_rows=heartbeat_every_rows,
            cpus_per_worker=cpus_per_worker,
            mem_mb_per_worker=mem_mb_per_worker,
            sync_local_dependencies=sync_local_dependencies,
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


def read_lerobot(
    root: str,
    *,
    fs: AbstractFileSystem | None = None,
    storage_options: Mapping[str, Any] | None = None,
    decode: bool = False,
) -> RefinerPipeline:
    """Create a pipeline with an episode-granular LeRobot reader source."""
    return RefinerPipeline(
        source=LeRobotEpisodeReader(
            root,
            fs=fs,
            storage_options=storage_options,
            decode=decode,
        )
    )


def from_items(
    items: Sequence[Any],
    *,
    shard_size_rows: int = 1_000,
) -> RefinerPipeline:
    """Create a pipeline from in-memory rows.

    Intended for small/medium inline datasets; large datasets should use file-backed
    readers (`read_parquet`/`read_jsonl`/`read_csv`). Primitive items are wrapped
    as ``{"item": value}``.
    """
    return RefinerPipeline(
        source=ItemsSource(
            items=items,
            shard_size_rows=shard_size_rows,
        )
    )


def from_source(source: BaseSource) -> RefinerPipeline:
    """Create a pipeline from a custom source implementation."""
    return RefinerPipeline(source=source)


def task(
    fn: Callable[[int, int], Any],
    *,
    num_tasks: int,
) -> RefinerPipeline:
    """Create a task-style pipeline with one callback invocation per rank."""
    source = TaskSource(num_tasks=num_tasks)
    return RefinerPipeline(source=source).add_step(
        FnRowStep(
            fn=lambda row: fn(row["task_rank"], num_tasks),
            index=1,
            op_name="task",
        )
    )


col = col_expr
