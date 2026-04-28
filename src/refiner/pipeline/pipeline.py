from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable

from fsspec import AbstractFileSystem

from refiner.io.datafolder import DataFolderLike
from refiner.pipeline.expressions import Expr, lit
from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.steps import (
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
    FnTableStep,
    MapFn,
    RenameStep,
    RefinerStep,
    SelectStep,
    VectorizedOp,
    VectorizedSegmentStep,
    WithColumnsStep,
)
from refiner.pipeline.sinks import BaseSink, JsonlSink, ParquetSink
from refiner.pipeline.sources import (
    BaseSource,
    CsvReader,
    HFDatasetReader,
    JsonlReader,
    ParquetReader,
)
from refiner.pipeline.sources.readers.lerobot import LeRobotEpisodeReader
from refiner.pipeline.sources.items import ItemsSource
from refiner.pipeline.sources.task import TaskSource
from refiner.pipeline.data.datatype import DTypeLike, DTypeMapping
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.shard import SHARD_ID_COLUMN
from refiner.execution.engine import (
    Block,
    Segment,
    compile_segments,
    execute_segments,
    iter_rows,
)
from refiner.execution.operators.row import ShardDeltaFn
from refiner.pipeline.sources.base import SourceUnit
from refiner.pipeline.sources.readers.utils import DEFAULT_TARGET_SHARD_BYTES
import pyarrow as pa

if TYPE_CHECKING:
    from refiner.launchers.cloud import CloudLaunchResult
    from refiner.launchers.local import LaunchStats


class RefinerPipeline:
    source: BaseSource
    pipeline_steps: tuple[RefinerStep, ...]
    _compiled_segments: tuple[Segment, ...] | None
    max_vectorized_block_bytes: int | None
    sink: BaseSink | None

    def __init__(
        self,
        source: BaseSource,
        pipeline_steps: Sequence[RefinerStep] | None = None,
        *,
        max_vectorized_block_bytes: int | None = None,
        sink: BaseSink | None = None,
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

    def _next_step_index(self) -> int:
        return 1 + sum(
            len(step.ops) if isinstance(step, VectorizedSegmentStep) else 1
            for step in self.pipeline_steps
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

    def with_sink(self, sink: BaseSink | None) -> "RefinerPipeline":
        return self.__class__(
            self.source,
            self.pipeline_steps,
            max_vectorized_block_bytes=self.max_vectorized_block_bytes,
            sink=sink,
        )

    def _get_compiled_segments(self) -> tuple[Segment, ...]:
        if self._compiled_segments is None:
            self._compiled_segments = compile_segments(self.pipeline_steps)
        return self._compiled_segments

    def map(
        self, fn: MapFn, *, dtypes: DTypeMapping | None = None
    ) -> "RefinerPipeline":
        return self.add_step(
            FnRowStep(
                fn=fn,
                op_name="map",
                index=self._next_step_index(),
                dtypes=dtypes,
            )
        )

    def map_async(
        self,
        fn: AsyncMapFn,
        *,
        max_in_flight: int = 16,
        preserve_order: bool = True,
        dtypes: DTypeMapping | None = None,
    ) -> "RefinerPipeline":
        return self.add_step(
            FnAsyncRowStep(
                fn=fn,
                max_in_flight=max_in_flight,
                preserve_order=preserve_order,
                op_name="map_async",
                index=self._next_step_index(),
                dtypes=dtypes,
            )
        )

    def batch_map(
        self,
        fn: BatchFn,
        *,
        batch_size: int,
        dtypes: DTypeMapping | None = None,
    ) -> "RefinerPipeline":
        if batch_size <= 1:
            raise ValueError("batch_size for batch_map must be > 1")
        return self.add_step(
            FnBatchStep(
                fn=fn,
                batch_size=batch_size,
                op_name="batch_map",
                index=self._next_step_index(),
                dtypes=dtypes,
            )
        )

    def flat_map(
        self,
        fn: FlatMapFn,
        *,
        dtypes: DTypeMapping | None = None,
    ) -> "RefinerPipeline":
        return self.add_step(
            FnFlatMapStep(
                fn=fn,
                op_name="flat_map",
                index=self._next_step_index(),
                dtypes=dtypes,
            )
        )

    def map_table(self, fn: Callable[[pa.Table], pa.Table]) -> "RefinerPipeline":
        return self._add_vectorized_op(
            FnTableStep(fn=fn, index=self._next_step_index())
        )

    def filter(self, predicate: Callable[[Row], bool] | Expr) -> "RefinerPipeline":
        if isinstance(predicate, Expr):
            return self._add_vectorized_op(
                FilterExprStep(predicate=predicate, index=self._next_step_index())
            )
        return self.add_step(
            FilterRowStep(
                predicate=predicate,
                op_name="filter",
                index=self._next_step_index(),
            )
        )

    def select(self, *columns: str) -> "RefinerPipeline":
        if not columns:
            raise ValueError("select requires at least one column")
        if SHARD_ID_COLUMN in columns:
            raise ValueError(f"{SHARD_ID_COLUMN} is an internal column")
        return self._add_vectorized_op(
            SelectStep(
                columns=tuple(columns) + (SHARD_ID_COLUMN,),
                index=self._next_step_index(),
            )
        )

    def with_columns(self, **assignments: Expr | Any) -> "RefinerPipeline":
        if not assignments:
            raise ValueError("with_columns requires at least one assignment")
        if SHARD_ID_COLUMN in assignments:
            raise ValueError(f"{SHARD_ID_COLUMN} is an internal column")
        exprs = {
            name: value if isinstance(value, Expr) else lit(value)
            for name, value in assignments.items()
        }
        return self._add_vectorized_op(
            WithColumnsStep(assignments=exprs, index=self._next_step_index())
        )

    def with_column(self, name: str, value: Expr | Any) -> "RefinerPipeline":
        if name == SHARD_ID_COLUMN:
            raise ValueError(f"{SHARD_ID_COLUMN} is an internal column")
        expr = value if isinstance(value, Expr) else lit(value)
        return self._add_vectorized_op(
            WithColumnsStep(assignments={name: expr}, index=self._next_step_index())
        )

    def drop(self, *columns: str) -> "RefinerPipeline":
        if not columns:
            raise ValueError("drop requires at least one column")
        if SHARD_ID_COLUMN in columns:
            raise ValueError(f"{SHARD_ID_COLUMN} is an internal column")
        return self._add_vectorized_op(
            DropStep(columns=tuple(columns), index=self._next_step_index())
        )

    def rename(self, **mapping: str) -> "RefinerPipeline":
        if not mapping:
            raise ValueError("rename requires at least one mapping")
        if SHARD_ID_COLUMN in mapping or SHARD_ID_COLUMN in mapping.values():
            raise ValueError(f"{SHARD_ID_COLUMN} is an internal column")
        return self._add_vectorized_op(
            RenameStep(mapping=mapping, index=self._next_step_index())
        )

    def cast(self, **dtypes: DTypeLike) -> "RefinerPipeline":
        if not dtypes:
            raise ValueError("cast requires at least one dtype mapping")
        if SHARD_ID_COLUMN in dtypes:
            raise ValueError(f"{SHARD_ID_COLUMN} is an internal column")
        return self._add_vectorized_op(
            CastStep(dtypes=dtypes, index=self._next_step_index())
        )

    def execute(
        self,
        rows: Iterable[SourceUnit],
        *,
        on_shard_delta: ShardDeltaFn | None = None,
    ) -> Iterable[Block]:
        """Execute source stream through compiled segments.

        Returns internal execution blocks (row blocks or tabular blocks).
        Use `iter_rows()` to force row iteration.
        """
        yield from execute_segments(
            rows,
            self._get_compiled_segments(),
            max_vectorized_block_bytes=self.max_vectorized_block_bytes,
            on_shard_delta=on_shard_delta,
            input_schema=self.source.schema,
            final_output_tabular=(
                self.sink.requires_tabular_input if self.sink is not None else False
            ),
        )

    def iter_rows(self) -> Iterable[Row]:
        """Local execution mode: lazily process all shards and yield output rows."""
        return iter_rows(self.execute(self.source.read()))

    def list_shards(self):
        return list(self.source.list_shards())

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

    def write_jsonl(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}__w{worker_id}.jsonl",
    ) -> "RefinerPipeline":
        return self.with_sink(
            JsonlSink(
                output=output,
                filename_template=filename_template,
            )
        )

    def write_parquet(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}__w{worker_id}.parquet",
        compression: str | None = None,
    ) -> "RefinerPipeline":
        return self.with_sink(
            ParquetSink(
                output=output,
                filename_template=filename_template,
                compression=compression,
            )
        )

    def __iter__(self) -> Iterator[Row]:
        return iter(self.iter_rows())

    def launch_local(
        self,
        *,
        name: str,
        num_workers: int = 1,
        rundir: str | None = None,
        gpus_per_worker: int | None = None,
    ) -> "LaunchStats":
        """Launch the pipeline locally.

        Args:
            name: Human-readable run name.
            num_workers: Number of local worker processes.
            rundir: Optional explicit local run directory. Reuse it to resume a prior local run.
            gpus_per_worker: Optional GPU devices exposed per worker.
        """
        from refiner.launchers.local import LocalLauncher

        launcher = LocalLauncher(
            pipeline=self,
            name=name,
            num_workers=num_workers,
            rundir=rundir,
            gpus_per_worker=gpus_per_worker,
        )
        return launcher.launch()

    def launch_cloud(
        self,
        *,
        name: str,
        num_workers: int = 1,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
        gpus_per_worker: int | None = None,
        gpu_type: str | None = None,
        sync_local_dependencies: bool = True,
        secrets: Mapping[str, object | None] | None = None,
        env: Mapping[str, object | None] | None = None,
        continue_from_job: str | None = None,
        unsafe_continue: bool = False,
    ) -> "CloudLaunchResult":
        """Launch the pipeline on Macrodata Cloud.

        Args:
            name: Human-readable run name.
            num_workers: Requested logical worker count.
            cpus_per_worker: Optional requested CPU cores per worker.
            mem_mb_per_worker: Optional requested memory in MB per worker for cloud scheduling.
            gpus_per_worker: Optional requested GPU count per worker for cloud scheduling.
            gpu_type: Optional requested GPU type per worker for cloud scheduling.
            sync_local_dependencies: Sync submitting environment dependencies in cloud image.
            secrets: Extra environment variables to mount inside the cloud image.
                `None` values are loaded from the submitting environment.
            env: Extra environment variables to mount inside the cloud image without
                treating their values as redaction targets. `None` values are loaded
                from the submitting environment.
            continue_from_job: Explicit continue selector. Accepts one prior cloud
                job id, one prior job id plus `:stage_index`, or `"infer"`.
            unsafe_continue: Allow continue when the reused stage boundary is not
                fully compatible with the current pipeline.
        """
        from refiner.launchers.cloud import CloudLauncher

        launcher = CloudLauncher(
            pipeline=self,
            name=name,
            num_workers=num_workers,
            cpus_per_worker=cpus_per_worker,
            mem_mb_per_worker=mem_mb_per_worker,
            gpus_per_worker=gpus_per_worker,
            gpu_type=gpu_type,
            sync_local_dependencies=sync_local_dependencies,
            secrets=dict(secrets) if secrets is not None else None,
            env=dict(env) if env is not None else None,
            continue_from_job=continue_from_job,
            unsafe_continue=unsafe_continue,
        )
        return launcher.launch()

    def write_lerobot(
        self,
        output: DataFolderLike,
        *,
        data_files_size_in_mb: int = 100,
        video_files_size_in_mb: int = 200,
        max_video_prepare_in_flight: int = 10,
        codec: str = "mpeg4",
        pix_fmt: str = "yuv420p",
        transencoding_threads: int | None = None,
        encoder_options: Mapping[str, str] | None = None,
        quantile_bins: int = 5000,
        force_recompute_video_stats: bool = False,
    ) -> "RefinerPipeline":
        """Append a deferred LeRobot writer sink and return a pipeline."""
        from refiner.pipeline.sinks.lerobot import LeRobotWriterSink

        return self.with_sink(
            LeRobotWriterSink(
                output=output,
                data_files_size_in_mb=data_files_size_in_mb,
                video_files_size_in_mb=video_files_size_in_mb,
                max_video_prepare_in_flight=max_video_prepare_in_flight,
                codec=codec,
                pix_fmt=pix_fmt,
                transencoding_threads=transencoding_threads,
                encoder_options=encoder_options,
                quantile_bins=quantile_bins,
                force_recompute_video_stats=force_recompute_video_stats,
            )
        )


## readers
def read_csv(
    inputs: DataFileSetLike,
    *,
    fs: AbstractFileSystem | None = None,
    storage_options: Mapping[str, Any] | None = None,
    recursive: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    num_shards: int | None = None,
    file_path_column: str | None = "file_path",
    multiline_rows: bool = False,
    encoding: str = "utf-8",
    parse_use_threads: bool = False,
    dtypes: DTypeMapping | None = None,
) -> RefinerPipeline:
    """Create a pipeline with a CSV reader source.

    `num_shards` and `target_shard_bytes` affect input shard planning on the
    reader side. `parse_use_threads` controls Arrow's intra-shard CSV parsing.
    """
    return RefinerPipeline(
        source=CsvReader(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            file_path_column=file_path_column,
            multiline_rows=multiline_rows,
            encoding=encoding,
            parse_use_threads=parse_use_threads,
            dtypes=dtypes,
        )
    )


def read_jsonl(
    inputs: DataFileSetLike,
    *,
    fs: AbstractFileSystem | None = None,
    storage_options: Mapping[str, Any] | None = None,
    recursive: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    num_shards: int | None = None,
    file_path_column: str | None = "file_path",
    parse_use_threads: bool = False,
    dtypes: DTypeMapping | None = None,
) -> RefinerPipeline:
    """Create a pipeline with a JSONL reader source.

    `num_shards` and `target_shard_bytes` affect input shard planning on the
    reader side. `parse_use_threads` controls Arrow's intra-shard JSON parsing.
    """
    return RefinerPipeline(
        source=JsonlReader(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            file_path_column=file_path_column,
            parse_use_threads=parse_use_threads,
            dtypes=dtypes,
        )
    )


def read_parquet(
    inputs: DataFileSetLike,
    *,
    fs: AbstractFileSystem | None = None,
    storage_options: Mapping[str, Any] | None = None,
    recursive: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    num_shards: int | None = None,
    arrow_batch_size: int = 65536,
    columns_to_read: Sequence[str] | None = None,
    filter: Expr | None = None,
    split_row_groups: bool = False,
    file_path_column: str | None = "file_path",
    dtypes: DTypeMapping | None = None,
) -> RefinerPipeline:
    """Create a pipeline with a Parquet reader source.

    `num_shards` and `target_shard_bytes` affect input shard planning on the
    reader side. Parquet always plans byte/file spans first and resolves them
    to row groups or row ranges at read time. `filter` uses Arrow expressions
    for row-group pruning plus row-level filtering during reads.
    """
    return RefinerPipeline(
        source=ParquetReader(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            arrow_batch_size=arrow_batch_size,
            columns_to_read=columns_to_read,
            filter=filter,
            split_row_groups=split_row_groups,
            file_path_column=file_path_column,
            dtypes=dtypes,
        )
    )


def read_hf_dataset(
    repo: str,
    config: str | None = None,
    split: str = "train",
    *,
    resolve_relative_paths: bool = True,
    dtypes: DTypeMapping | None = None,
    hf_token: str | None = None,
    timeout: float = 30.0,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    num_shards: int | None = None,
    arrow_batch_size: int = 65536,
    columns_to_read: Sequence[str] | None = None,
    filter: Expr | None = None,
    split_row_groups: bool = False,
    file_path_column: str | None = "file_path",
) -> RefinerPipeline:
    """Create a pipeline over Hugging Face dataset parquet shards.

    Args:
        repo: Hugging Face dataset repository ID.
        config: Dataset config name. If omitted, Hugging Face datasets resolves it.
        split: Dataset split to read.
        resolve_relative_paths: Whether file-typed relative paths should be rewritten
            as `hf://datasets/{repo}/...` references. Absolute paths and URI values
            are left unchanged.
    """
    return RefinerPipeline(
        source=HFDatasetReader(
            repo,
            config=config,
            split=split,
            resolve_relative_paths=resolve_relative_paths,
            dtypes=dtypes,
            hf_token=hf_token,
            timeout=timeout,
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            arrow_batch_size=arrow_batch_size,
            columns_to_read=columns_to_read,
            filter=filter,
            split_row_groups=split_row_groups,
            file_path_column=file_path_column,
        )
    )


def read_lerobot(
    inputs: DataFolderLike | Sequence[DataFolderLike],
    *,
    fs: AbstractFileSystem | None = None,
    storage_options: Mapping[str, Any] | None = None,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    num_shards: int | None = None,
    split_row_groups: bool = True,
) -> RefinerPipeline:
    """Create a pipeline with an episode-granular LeRobot reader source.

    `inputs` is the LeRobot dataset root.
    `num_shards` and `target_shard_bytes` affect only episode parquet shard
    planning. `split_row_groups` controls whether those planned spans are
    refined to row ranges inside an episode parquet file. Media loading stays
    unchanged.
    """
    return RefinerPipeline(
        source=LeRobotEpisodeReader(
            inputs,
            fs=fs,
            storage_options=storage_options,
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            split_row_groups=split_row_groups,
        )
    )


def from_items(
    items: Sequence[Any],
    *,
    items_per_shard: int = 1_000,
) -> RefinerPipeline:
    """Create a pipeline from in-memory rows.

    Intended for small/medium inline datasets; large datasets should use file-backed
    readers (`read_parquet`/`read_jsonl`/`read_csv`). Primitive items are wrapped
    as ``{"item": value}``.
    """
    return RefinerPipeline(
        source=ItemsSource(
            items=items,
            items_per_shard=items_per_shard,
        )
    )


def from_source(source: BaseSource) -> RefinerPipeline:
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
