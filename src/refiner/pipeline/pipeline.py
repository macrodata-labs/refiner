from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, cast

from fsspec import AbstractFileSystem

from refiner.io.datafolder import DataFolderLike
from refiner.pipeline.expressions import Expr, lit
from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.resources import GPU
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
from refiner.pipeline.sinks import BaseSink, JsonlSink, ParquetSink, ZarrSink
from refiner.pipeline.sinks.assets import MissingAssetPolicy
from refiner.pipeline.sources import (
    BaseSource,
    CsvReader,
    FilesReader,
    HFDatasetReader,
    Hdf5Reader,
    JsonReader,
    McapReader,
    ParquetReader,
    TfdsReader,
    TfrecordReader,
    ZarrReader,
)
from refiner.pipeline.sources.readers.hdf5 import MissingPolicy
from refiner.pipeline.sources.readers.lerobot import LeRobotEpisodeReader
from refiner.pipeline.sources.readers.mcap import SyncMethod
from refiner.pipeline.sources.items import ItemsSource
from refiner.pipeline.sources.task import TaskSource, TaskStep
from refiner.pipeline.data import datatype
from refiner.pipeline.data.datatype import DTypeLike, DTypeMapping
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.shard import SHARD_ID_COLUMN
from refiner.execution.engine import (
    Block,
    Segment,
    compile_segments,
    execute_segments,
    iter_rows,
    schema_after_segments,
)
from refiner.execution.operators.row import ShardDeltaFn
from refiner.pipeline.sources.base import SourceUnit
from refiner.pipeline.sources.readers.utils import (
    DEFAULT_TARGET_SHARD_BYTES,
    PathSelection,
)
import pyarrow as pa

_DEFAULT_LEROBOT_ENCODER_OPTIONS: Mapping[str, str] = {"g": "2"}

if TYPE_CHECKING:
    from refiner.launchers.cloud import CloudLaunchResult
    from refiner.launchers.local import LaunchStats
    from refiner.launchers.secrets import SecretInput


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
        """Create an immutable pipeline value.

        Args:
            source: Source that plans shards and emits source rows or blocks.
            pipeline_steps: Ordered transform steps applied after the source.
            max_vectorized_block_bytes: Optional target byte cap for vectorized
                Arrow blocks. Smaller values reduce peak memory at the cost of
                more block boundaries.
            sink: Optional writer sink attached by a ``write_*`` method.
        """
        if max_vectorized_block_bytes is not None and max_vectorized_block_bytes <= 0:
            raise ValueError("max_vectorized_block_bytes must be > 0 when provided")
        self.source = source
        self.pipeline_steps = tuple(pipeline_steps) if pipeline_steps else ()
        self._compiled_segments = None
        self.max_vectorized_block_bytes = max_vectorized_block_bytes
        self.sink = sink

    def add_step(self, step: RefinerStep) -> "RefinerPipeline":
        """Return a new pipeline with one transform step appended.

        Pipelines are immutable: this method does not mutate the current
        instance and preserves the current source, sink, and vectorized block
        settings.
        """
        return self.__class__(
            self.source,
            self.pipeline_steps + (step,),
            max_vectorized_block_bytes=self.max_vectorized_block_bytes,
            sink=self.sink,
        )

    def _next_step_index(self) -> int:
        """Return the user-visible index assigned to the next operation.

        Vectorized fused segments can contain multiple logical operations, so
        this counts individual operations rather than raw pipeline step objects.
        """
        return 1 + sum(
            len(step.ops) if isinstance(step, VectorizedSegmentStep) else 1
            for step in self.pipeline_steps
        )

    def _add_vectorized_op(self, op: VectorizedOp) -> "RefinerPipeline":
        """Append an Arrow/vectorized operation, fusing adjacent vectorized ops.

        Fusing keeps expression-backed transforms inside one Arrow execution
        boundary, avoiding avoidable row materialization between compatible
        operations.
        """
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
        """Return a copy with a different vectorized block byte cap.

        Set this when vectorized expression/table operations should operate on
        smaller Arrow chunks to reduce memory pressure. ``None`` leaves block
        sizing to the execution engine.
        """
        return self.__class__(
            self.source,
            self.pipeline_steps,
            max_vectorized_block_bytes=max_vectorized_block_bytes,
            sink=self.sink,
        )

    def with_sink(self, sink: BaseSink | None) -> "RefinerPipeline":
        """Return a copy with the given sink attached or removed.

        Writer helpers such as ``write_jsonl`` and ``write_lerobot`` call this
        method. Passing ``None`` removes the sink and leaves a read/transform
        pipeline suitable for inspection.
        """
        return self.__class__(
            self.source,
            self.pipeline_steps,
            max_vectorized_block_bytes=self.max_vectorized_block_bytes,
            sink=sink,
        )

    def _get_compiled_segments(self) -> tuple[Segment, ...]:
        """Compile and cache execution segments for the current step sequence.

        Compilation groups compatible steps into executable segments. The result
        is cached because pipelines are immutable after construction.
        """
        if self._compiled_segments is None:
            self._compiled_segments = compile_segments(self.pipeline_steps)
        return self._compiled_segments

    def output_schema(self) -> pa.Schema | None:
        """Return the best-known Arrow schema after vectorized transforms.

        The result can be ``None`` when the source or row-level transforms do
        not expose a static schema. Row-level Python callbacks may still emit
        fields not visible here unless they declare ``dtypes``.
        """
        return schema_after_segments(self.source.schema, self._get_compiled_segments())

    def map(
        self, fn: MapFn, *, dtypes: DTypeMapping | None = None
    ) -> "RefinerPipeline":
        """Apply a Python function to each row.

        ``fn`` receives one ``Row`` and may return a replacement row, a mapping
        patch, or another row-like object accepted by the execution engine.
        Provide ``dtypes`` when the function creates or changes columns and you
        want downstream vectorized operations or writers to know the schema.
        """
        return self.add_step(
            FnRowStep(
                fn=fn,
                op_name="map",
                index=self._next_step_index(),
                dtypes=dtypes,
            )
        )

    def to_robot_rows(
        self,
        *,
        episode_id_key: str | None = None,
        task_key: str | None = None,
        fps: float | None = None,
        fps_key: str | None = None,
        robot_type: str | None = None,
        robot_type_key: str | None = None,
        nested_frames_key: str | None = None,
        timestamp_key: str | None = "timestamp",
        action_key: str | None = "action",
        state_key: str | Sequence[str] | None = "observation.state",
        extra_observation_keys: Mapping[str, str] | Iterable[str] | None = None,
        video_keys: Mapping[str, str] | Iterable[str] | None = None,
        stats_key: str | None = "stats",
        stats_prefix: str = "stats/",
    ) -> "RefinerPipeline":
        """Expose rows through the RoboticsRow semantic view.

        This does not materialize semantic properties such as ``episode_id`` or
        ``num_frames`` as physical table columns. Vectorized operations after this
        step still address the underlying source columns, so use the original key
        names in expressions unless you have explicitly created new columns. Row-level
        operations and iteration can still access the semantic properties directly:
        ``.filter(lambda row: row.episode_id == "ep-1")`` uses the view property,
        while ``.filter(col("episode_id") == "ep-1")`` requires a physical
        ``episode_id`` column.

        ``episode_id_key`` and ``task_key`` may use slash paths. ``task_key`` values
        are exposed as ``row.tasks``; strings become single-item task lists, and
        sequences of strings are preserved. For nested frame rows, ``task_key`` may
        point inside ``nested_frames_key``. If ``timestamp_key`` is requested but
        absent and ``fps`` is known, timestamps are generated from frame indices.
        """
        from refiner.robotics.row import _robot_row_converter

        converter = _robot_row_converter(
            episode_id_key=episode_id_key,
            task_key=task_key,
            fps=fps,
            fps_key=fps_key,
            robot_type=robot_type,
            robot_type_key=robot_type_key,
            nested_frames_key=nested_frames_key,
            timestamp_key=timestamp_key,
            action_key=action_key,
            state_key=state_key,
            extra_observation_keys=extra_observation_keys,
            video_keys=video_keys,
            schema=self.output_schema(),
            stats_key=stats_key,
            stats_prefix=stats_prefix,
        )
        return self.map(cast(MapFn, converter))

    def map_async(
        self,
        fn: AsyncMapFn,
        *,
        max_in_flight: int = 16,
        preserve_order: bool = True,
        dtypes: DTypeMapping | None = None,
    ) -> "RefinerPipeline":
        """Apply an async Python function to each row.

        Args:
            fn: Async callback receiving one row.
            max_in_flight: Maximum unresolved callback tasks per worker.
            preserve_order: If True, emit rows in input order. If False, emit
                rows as callbacks finish.
            dtypes: Optional dtype/schema hints for fields produced by ``fn``.
        """
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
        """Apply a Python function to fixed-size row batches.

        ``fn`` receives batches of rows and returns rows or row patches according
        to the batch transform contract. Use this for APIs that are more
        efficient when called on multiple rows at once.
        """
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
        """Map each input row to zero or more output rows.

        Use ``flat_map`` for expansion, filtering with emitted replacements, or
        splitting one source row into several logical rows. Provide ``dtypes``
        when the callback changes the output schema.
        """
        return self.add_step(
            FnFlatMapStep(
                fn=fn,
                op_name="flat_map",
                index=self._next_step_index(),
                dtypes=dtypes,
            )
        )

    def map_table(self, fn: Callable[[pa.Table], pa.Table]) -> "RefinerPipeline":
        """Apply a vectorized Arrow table transform.

        ``fn`` receives a ``pyarrow.Table`` and must return a ``pyarrow.Table``.
        Adjacent vectorized operations are fused so they can run inside the same
        Arrow segment.
        """
        return self._add_vectorized_op(
            FnTableStep(fn=fn, index=self._next_step_index())
        )

    def filter(self, predicate: Callable[[Row], bool] | Expr) -> "RefinerPipeline":
        """Keep rows matching a Python predicate or vectorized expression.

        A Python callable receives one row at a time. An ``Expr`` runs as a
        vectorized Arrow filter and can be fused with adjacent vectorized
        operations.
        """
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
        """Keep only the named columns.

        The internal shard id column is preserved automatically for execution
        bookkeeping and is not part of the public column selection.
        """
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
        """Add or replace columns with vectorized expressions or literals.

        Keyword names are output columns. Values that are not ``Expr`` instances
        are treated as literals and broadcast to each row in the current block.
        """
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
        """Add or replace one column with a vectorized expression or literal.

        This is a convenience wrapper around ``with_columns`` for a single
        assignment. Non-expression values are treated as literals.
        """
        if name == SHARD_ID_COLUMN:
            raise ValueError(f"{SHARD_ID_COLUMN} is an internal column")
        expr = value if isinstance(value, Expr) else lit(value)
        return self._add_vectorized_op(
            WithColumnsStep(assignments={name: expr}, index=self._next_step_index())
        )

    def drop(self, *columns: str) -> "RefinerPipeline":
        """Drop the named columns from each row or Arrow block.

        ``drop`` is vectorized and can be fused with adjacent expression-backed
        operations. The internal shard id column cannot be dropped.
        """
        if not columns:
            raise ValueError("drop requires at least one column")
        if SHARD_ID_COLUMN in columns:
            raise ValueError(f"{SHARD_ID_COLUMN} is an internal column")
        return self._add_vectorized_op(
            DropStep(columns=tuple(columns), index=self._next_step_index())
        )

    def rename(self, **mapping: str) -> "RefinerPipeline":
        """Rename columns using ``old_name=new_name`` keyword arguments.

        For example, ``pipeline.rename(old="new")`` renames column ``old`` to
        ``new``. The internal shard id column cannot be renamed.
        """
        if not mapping:
            raise ValueError("rename requires at least one mapping")
        if SHARD_ID_COLUMN in mapping or SHARD_ID_COLUMN in mapping.values():
            raise ValueError(f"{SHARD_ID_COLUMN} is an internal column")
        return self._add_vectorized_op(
            RenameStep(mapping=mapping, index=self._next_step_index())
        )

    def cast(self, **dtypes: DTypeLike) -> "RefinerPipeline":
        """Cast columns to Arrow/refiner dtypes.

        Keyword names are column names and values are dtype specifications
        accepted by ``refiner.pipeline.data.datatype``.
        """
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

        This is the low-level execution primitive used by workers and local
        inspection helpers. It returns internal row or table blocks rather than
        forcing materialization as rows. Most user code should prefer
        ``iter_rows()``, ``take()``, or a launcher.

        Args:
            rows: Source units from ``source.read()`` or a compatible stream.
            on_shard_delta: Optional callback used by workers to track shard
                progress as rows move through the pipeline.
        """
        yield from execute_segments(
            rows,
            self._get_compiled_segments(),
            max_vectorized_block_bytes=self.max_vectorized_block_bytes,
            on_shard_delta=on_shard_delta,
            input_schema=self.source.schema,
        )

    def iter_rows(self) -> Iterable[Row]:
        """Lazily execute the pipeline locally and yield rows.

        This is an in-process inspection path. It does not launch worker
        processes and does not run attached sinks; use ``launch_local`` or
        ``launch_cloud`` to execute writers.
        """
        return iter_rows(self.execute(self.source.read()))

    def list_shards(self):
        """Return the source shards that would be processed by a launch.

        This delegates to the source shard planner and is useful for debugging
        sharding decisions without executing the pipeline transforms.
        """
        return list(self.source.list_shards())

    def materialize(self) -> list[Row]:
        """Execute locally and collect every output row into memory.

        This is intended for small local/debug workloads. For large datasets,
        prefer ``iter_rows()``, ``take()``, or launched execution.
        """
        return list(self.iter_rows())

    def take(self, n: int) -> list[Row]:
        """Return up to the first ``n`` rows from local execution.

        ``take`` stops reading as soon as enough rows have been produced, making
        it the preferred way to inspect schemas, media references, and transform
        outputs before launching a full pipeline.
        """
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
        upload_assets: bool = False,
        assets_subdir: str = "assets",
        max_asset_uploads_in_flight: int = 16,
        missing_asset_policy: MissingAssetPolicy = "error",
    ) -> "RefinerPipeline":
        """Attach a JSONL writer sink.

        Args:
            output: Output folder or URL prefix.
            filename_template: Per-worker output filename template. Available
                fields include ``shard_id`` and ``worker_id``.
            upload_assets: Whether referenced local assets should be copied
                beside the JSONL output and rewritten to output paths.
            assets_subdir: Subdirectory used when ``upload_assets`` is enabled.
            max_asset_uploads_in_flight: Concurrent asset uploads per worker.
            missing_asset_policy: How missing assets are handled when uploading:
                error, keep the original reference, or write null depending on
                the sink policy.
        """
        return self.with_sink(
            JsonlSink(
                output=output,
                filename_template=filename_template,
                upload_assets=upload_assets,
                assets_subdir=assets_subdir,
                max_asset_uploads_in_flight=max_asset_uploads_in_flight,
                missing_asset_policy=missing_asset_policy,
            )
        )

    def write_parquet(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}__w{worker_id}.parquet",
        compression: str | None = None,
        upload_assets: bool = False,
        assets_subdir: str = "assets",
        max_asset_uploads_in_flight: int = 16,
        missing_asset_policy: MissingAssetPolicy = "error",
        dtypes: DTypeMapping | None = None,
    ) -> "RefinerPipeline":
        """Attach a Parquet writer sink.

        Args:
            output: Output folder or URL prefix.
            filename_template: Per-worker output filename template. Available
                fields include ``shard_id`` and ``worker_id``.
            compression: Optional Parquet compression codec.
            upload_assets: Whether referenced local assets should be copied
                beside the Parquet output and rewritten to output paths.
            assets_subdir: Subdirectory used when ``upload_assets`` is enabled.
            max_asset_uploads_in_flight: Concurrent asset uploads per worker.
            missing_asset_policy: How missing assets are handled when uploading.
            dtypes: Optional dtype overrides for written columns.
        """
        return self.with_sink(
            ParquetSink(
                output=output,
                filename_template=filename_template,
                compression=compression,
                upload_assets=upload_assets,
                assets_subdir=assets_subdir,
                max_asset_uploads_in_flight=max_asset_uploads_in_flight,
                missing_asset_policy=missing_asset_policy,
                dtypes=dtypes,
            )
        )

    def write_zarr(
        self,
        output: DataFolderLike,
        *,
        arrays: Mapping[str, str] | None = None,
        attrs: Mapping[str, str] | None = None,
        episode_ends_path: str | None = "meta/episode_ends",
        store_template: str = "{shard_id}__w{worker_id}.zarr",
        video_frame_batch_size: int = 8,
        array_chunk_bytes: int = 8 * 1024 * 1024,
        reduce_to_single_store: bool = True,
    ) -> "RefinerPipeline":
        """Write rows to Zarr array stores.

        Args:
            output: Output folder or URL prefix for the Zarr store(s).
            arrays: Mapping from output Zarr array path to source row key. If
                omitted for ``RoboticsRow`` inputs, writes the available default
                robotics arrays: actions, states, timestamps, and videos.
            attrs: Mapping from output Zarr root attribute name to source row key.
                Attribute values must be stable across rows in each output store.
            episode_ends_path: Output Zarr path for cumulative row/episode end
                offsets. Set to None to omit episode boundaries.
            store_template: Per-shard store path template. Must include
                ``{shard_id}`` and ``{worker_id}``.
            video_frame_batch_size: Maximum decoded video frames to append per
                video write batch.
            array_chunk_bytes: Target byte size for chunks created for newly
                written arrays and for read/write batches when reducing shard
                stores into a single store.
            reduce_to_single_store: If True, add a reducer stage that merges
                shard-local stores into one Zarr group at ``output``. Defaults
                to True.
        """
        return self.with_sink(
            ZarrSink(
                output=output,
                arrays=arrays,
                attrs=attrs,
                episode_ends_path=episode_ends_path,
                store_template=store_template,
                video_frame_batch_size=video_frame_batch_size,
                array_chunk_bytes=array_chunk_bytes,
                reduce_to_single_store=reduce_to_single_store,
            )
        )

    def __iter__(self) -> Iterator[Row]:
        """Iterate over rows using in-process local execution.

        This makes ``for row in pipeline`` equivalent to ``pipeline.iter_rows()``.
        Attached sinks are not executed by iteration.
        """
        return iter(self.iter_rows())

    def launch_local(
        self,
        *,
        name: str,
        num_workers: int = 1,
        rundir: str | None = None,
        gpu: GPU | None = None,
    ) -> "LaunchStats":
        """Launch the pipeline locally.

        Args:
            name: Human-readable run name.
            num_workers: Number of local worker processes.
            rundir: Optional explicit local run directory. Reuse it to resume a prior local run.
            gpu: Optional GPU devices exposed per worker. `cuda_version` is accepted
                for API consistency but ignored by local launch.
        """
        from refiner.launchers.local import LocalLauncher

        launcher = LocalLauncher(
            pipeline=self,
            name=name,
            num_workers=num_workers,
            rundir=rundir,
            gpu=gpu,
        )
        return launcher.launch()

    def launch_cloud(
        self,
        *,
        name: str,
        num_workers: int = 1,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
        gpu: GPU | None = None,
        sync_local_dependencies: bool = False,
        dependencies: Sequence[str] | None = None,
        refiner_extras: Sequence[str] | None = None,
        secrets: SecretInput | None = None,
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
            gpu: Optional structured GPU request.
            sync_local_dependencies: Include packages detected from the local
                environment in the cloud runtime.
            dependencies: Additional packages to install in the cloud runtime.
                Entries are requirement strings such as `"torch"` or
                `"ego-vision[models]==0.1.2"`.
            refiner_extras: Additional macrodata-refiner extras to install in
                the cloud runtime. Built-in blocks automatically declare the
                extras they require; pass this for extras used outside those
                blocks.
            secrets: Secret sources to mount inside the cloud image. A mapping keeps
                the legacy behavior; `None` values are loaded from the submitting
                environment. `Secrets.env(...)` references stored workspace secrets.
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
            gpu=gpu,
            sync_local_dependencies=sync_local_dependencies,
            dependencies=dependencies,
            refiner_extras=refiner_extras,
            secrets=secrets,
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
        max_video_prepare_in_flight: int = 4,
        codec: str = "mpeg4",
        pix_fmt: str = "yuv420p",
        transencoding_threads: int | None = None,
        encoder_options: Mapping[str, str] | None = _DEFAULT_LEROBOT_ENCODER_OPTIONS,
        quantile_bins: int = 5000,
        force_recompute_video_stats: bool = False,
    ) -> "RefinerPipeline":
        """Append a LeRobot writer sink.

        The writer expects ``LeRobotRow`` or ``RoboticsRow`` inputs. It writes
        LeRobot frame parquet files, episode metadata, task metadata, video
        files, and aggregate statistics. Launched execution adds a reducer stage
        that finalizes global metadata after shard-local writes complete.

        Args:
            output: LeRobot output dataset root.
            data_files_size_in_mb: Target frame parquet file size.
            video_files_size_in_mb: Target video file size.
            max_video_prepare_in_flight: Bound concurrent episode video
                preparation per worker. Lower values reduce remote-video load
                and memory pressure.
            codec: Codec used when videos must be transcoded.
            pix_fmt: Pixel format used when videos must be transcoded.
            transencoding_threads: Optional encoder thread count.
            encoder_options: Optional codec-specific encoder options.
            quantile_bins: Accuracy/cost tradeoff for video stats quantiles.
            force_recompute_video_stats: Recompute video stats even when source
                LeRobot stats are available.
        """
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


def read_json(
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
    lines: bool = False,
) -> RefinerPipeline:
    """Create a pipeline with a JSON reader source.

    `num_shards` and `target_shard_bytes` affect input shard planning on the
    reader side. `parse_use_threads` controls Arrow's intra-shard JSON parsing
    when `lines=True`.
    """
    return RefinerPipeline(
        source=JsonReader(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            file_path_column=file_path_column,
            parse_use_threads=parse_use_threads,
            dtypes=dtypes,
            lines=lines,
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
    """Create a pipeline with a JSON Lines reader source.

    This is equivalent to ``read_json(..., lines=True)`` and exposes the same
    sharding, parsing, file path, and dtype options.
    """
    return read_json(
        inputs,
        fs=fs,
        storage_options=storage_options,
        recursive=recursive,
        target_shard_bytes=target_shard_bytes,
        num_shards=num_shards,
        file_path_column=file_path_column,
        parse_use_threads=parse_use_threads,
        dtypes=dtypes,
        lines=True,
    )


def read_files(
    inputs: DataFileSetLike,
    *,
    fs: AbstractFileSystem | None = None,
    storage_options: Mapping[str, Any] | None = None,
    recursive: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    num_shards: int | None = None,
    file_path_column: str | None = "file_path",
    content_column: str | None = None,
    size_column: str | None = "size",
    decode_fn: Callable[[bytes], Any] | None = None,
    max_in_flight: int = 8,
    dtypes: DTypeMapping | None = None,
) -> RefinerPipeline:
    """Create a pipeline that emits one row per resolved file.

    If `content_column` is set, each row includes the file's raw bytes in that
    column. Pass `decode_fn` to transform those bytes before they are emitted.
    Otherwise files are listed without opening their contents.
    If `size_column` is set, each row includes the file size captured during
    shard planning.

    Args:
        inputs: File, glob, directory, or sequence of fsspec-backed inputs.
        fs: Optional filesystem for string inputs.
        storage_options: fsspec options used when `fs` is not provided.
        recursive: Whether directory inputs are listed recursively.
        target_shard_bytes: Approximate target bytes per planned shard.
        num_shards: Optional requested number of planned shards.
        file_path_column: Path output column, or `None` to omit it.
        content_column: Raw bytes output column, or `None` for path-only rows.
        size_column: File size output column, or `None` to omit it.
        decode_fn: Optional function applied to raw file bytes when reading content.
        max_in_flight: Concurrent content reads per shard when reading bytes.
        dtypes: Optional dtype overrides exposed through the source schema.
    """
    return RefinerPipeline(
        source=FilesReader(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            file_path_column=file_path_column,
            content_column=content_column,
            size_column=size_column,
            decode_fn=decode_fn,
            max_in_flight=max_in_flight,
            dtypes=dtypes,
        )
    )


def read_videos(
    inputs: DataFileSetLike,
    *,
    fs: AbstractFileSystem | None = None,
    storage_options: Mapping[str, Any] | None = None,
    recursive: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    num_shards: int | None = None,
    file_path_column: str | None = "video_path",
    size_column: str | None = "size",
    dtypes: DTypeMapping | None = None,
) -> RefinerPipeline:
    """Create a path-only file reader whose path column is marked as video.

    This is a convenience wrapper around `read_files` that defaults the path
    column to `"video_path"` and marks it with `mdr.datatype.video_path()`.

    Args:
        inputs: Video file, glob, directory, or sequence of fsspec-backed inputs.
        fs: Optional filesystem for string inputs.
        storage_options: fsspec options used when `fs` is not provided.
        recursive: Whether directory inputs are listed recursively.
        target_shard_bytes: Approximate target bytes per planned shard.
        num_shards: Optional requested number of planned shards.
        file_path_column: Path output column, or `None` to omit it.
        size_column: File size output column, or `None` to omit it.
        dtypes: Optional dtype overrides exposed through the source schema.
    """
    video_dtypes = dict(dtypes or {})
    if file_path_column is not None and file_path_column not in video_dtypes:
        video_dtypes[file_path_column] = datatype.video_path()
    return read_files(
        inputs,
        fs=fs,
        storage_options=storage_options,
        recursive=recursive,
        target_shard_bytes=target_shard_bytes,
        num_shards=num_shards,
        file_path_column=file_path_column,
        size_column=size_column,
        dtypes=video_dtypes,
    )


def read_hdf5(
    inputs: DataFileSetLike,
    *,
    fs: AbstractFileSystem | None = None,
    storage_options: Mapping[str, Any] | None = None,
    recursive: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    num_shards: int | None = None,
    groups: str | Sequence[str] | None = None,
    datasets: PathSelection | None = None,
    attrs: PathSelection | None = None,
    file_path_column: str | None = "file_path",
    group_path_column: str | None = "hdf5_group",
    missing_policy: MissingPolicy = "error",
    cache_remote_files: bool = False,
    dtypes: DTypeMapping | None = None,
) -> RefinerPipeline:
    """Create a pipeline with an HDF5 reader source.

    HDF5 files are planned as atomic files. Each matched HDF5 group becomes one
    row, with `datasets` and `attrs` selecting values relative to that group.

    Args:
        inputs: HDF5 file, glob, directory, or list of files to read.
        fs: Optional fsspec filesystem for resolving inputs.
        storage_options: Optional fsspec storage options.
        recursive: Whether directory inputs should be expanded recursively.
        target_shard_bytes: Target shard size used when planning file shards.
        num_shards: Requested number of planned shards. HDF5 files are atomic,
            so readers may emit fewer shards when there are fewer files.
        groups: HDF5 group selector to emit as rows. Accepts `"/"`, one glob
            string, or a sequence of exact group paths. Defaults to the root group.
        datasets: Dataset selections relative to each matched group. Accepts a
            mapping of output column name to HDF5 path, one path string, or a
            sequence of path strings with unique final components.
        attrs: Attribute selections on each matched group. Accepts the same
            forms as `datasets`.
        file_path_column: Output column for the source file path, or `None` to
            omit it.
        group_path_column: Output column for the matched HDF5 group path, or
            `None` to omit it.
        missing_policy: How to handle missing selected datasets or attrs inside
            matched groups: `"error"` raises, `"drop_row"` drops the matched
            group row, and `"set_null"` keeps the row with nulls for missing
            selected values.
        cache_remote_files: If True, copy non-local HDF5 files to worker-local
            temporary storage before opening them. This can reduce random-access
            reads against object stores at the cost of downloading each file once.
        dtypes: Optional dtype overrides for output columns.
    """
    return RefinerPipeline(
        source=Hdf5Reader(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            groups=groups,
            datasets=datasets,
            attrs=attrs,
            file_path_column=file_path_column,
            group_path_column=group_path_column,
            missing_policy=missing_policy,
            cache_remote_files=cache_remote_files,
            dtypes=dtypes,
        )
    )


def read_zarr(
    input: DataFileSetLike,
    *,
    arrays: PathSelection | None = None,
    attrs: PathSelection | None = None,
    row_ends: str | None = None,
    split_leading_axis: bool = False,
    leading_axis_row_size: int = 1,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    num_shards: int | None = None,
    row_batch_size: int | None = None,
    index_column: str | None = "index",
    file_path_column: str | None = "file_path",
    dtypes: DTypeMapping | None = None,
) -> RefinerPipeline:
    """Create a pipeline with a Zarr reader source.

    The reader has three modes:
    - group mode: one Zarr group becomes one row
    - row_ends mode: cumulative offsets define whole-row source slices
    - split_leading_axis mode: fixed-size leading-axis slices define output rows

    Missing selected arrays or attributes raise immediately. `row_ends` and
    `split_leading_axis` are mutually exclusive. `target_shard_bytes` and
    `num_shards` affect shard planning, not logical row size. `row_batch_size`
    bounds how many logical rows are loaded per array block within each shard.

    Args:
        input: Zarr group path, glob, or sequence of Zarr group paths.
    """
    return RefinerPipeline(
        source=ZarrReader(
            input,
            arrays=arrays,
            attrs=attrs,
            row_ends=row_ends,
            split_leading_axis=split_leading_axis,
            leading_axis_row_size=leading_axis_row_size,
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            row_batch_size=row_batch_size,
            index_column=index_column,
            file_path_column=file_path_column,
            dtypes=dtypes,
        )
    )


def read_mcap(
    inputs: DataFileSetLike,
    *,
    fs: AbstractFileSystem | None = None,
    storage_options: Mapping[str, Any] | None = None,
    recursive: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    num_shards: int | None = None,
    file_path_column: str | None = "file_path",
    episode_splitting: str | Mapping[str, Any] = "single",
    stream_episodes: bool = False,
    assume_log_time_order: bool = False,
    fields: PathSelection | None = None,
    videos: PathSelection | None = None,
    sync_primary: str | None = None,
    sync_method: SyncMethod = "nearest",
    include_skew: bool = True,
    fps: float | None = None,
) -> RefinerPipeline:
    """Create a pipeline with an MCAP reader source.

    Each emitted row represents one episode and contains a `records` `Tabular`
    table. When `videos` is set, rows also include a `videos` mapping of
    selected names to video source values. Set `sync_primary` to align
    records and videos to a field, topic, or video timeline; otherwise selected
    fields are emitted sparsely at their original MCAP log timestamps.

    Args:
        inputs: MCAP file, glob, directory, or sequence of inputs.
        fs: Optional fsspec filesystem for string inputs.
        storage_options: Optional fsspec storage options.
        recursive: Whether directory inputs should be expanded recursively.
        target_shard_bytes: Target shard size used when planning files.
        num_shards: Requested number of planned shards. MCAP files are atomic,
            so readers may emit fewer shards when there are fewer files.
        file_path_column: Output column for the source file path, or `None` to
            omit it.
        episode_splitting: `"single"`, `{"time_gap_s": seconds}`, or
            `{"marker_topic": topic}`.
        stream_episodes: When splitting episodes, buffer one episode at a time
            for seekable indexed MCAPs.
        assume_log_time_order: When `stream_episodes=True`, stream
            non-seekable or unindexed files in physical file order instead of
            falling back, assuming messages are already ordered by log_time.
        fields: Record-table selections as output-name to MCAP source mapping,
            a single source string, a source sequence, or `None` to derive
            default fields from decoded non-video messages.
        videos: Video selections as video-name to MCAP source mapping, a single
            source string, a source sequence, or `None`.
        sync_primary: Optional selected field/video name, topic, or dotted MCAP
            source that defines aligned record timestamps.
        sync_method: Alignment method for non-primary fields and videos:
            `"nearest"`, `"hold"`, or `"interpolate"`.
        include_skew: Whether to add alignment timestamp/skew columns for
            non-primary aligned fields.
        fps: Positive explicit video/frame rate. If omitted, aligned reads infer
            it from `sync_primary` timestamps when possible.
    """
    return RefinerPipeline(
        source=McapReader(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            file_path_column=file_path_column,
            episode_splitting=episode_splitting,
            stream_episodes=stream_episodes,
            assume_log_time_order=assume_log_time_order,
            fields=fields,
            videos=videos,
            sync_primary=sync_primary,
            sync_method=sync_method,
            include_skew=include_skew,
            fps=fps,
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
    dtypes: DTypeMapping | None = None,
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
    """
    return RefinerPipeline(
        source=HFDatasetReader(
            repo,
            config=config,
            split=split,
            dtypes=dtypes,
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
    skip_malformed_rows: bool = False,
    limit: int | None = None,
) -> RefinerPipeline:
    """Create a pipeline with an episode-granular LeRobot reader source.

    `inputs` is the LeRobot dataset root.
    `num_shards` and `target_shard_bytes` affect only episode parquet shard
    planning. `split_row_groups` controls whether those planned spans are
    refined to row ranges inside an episode parquet file. Media loading stays
    unchanged. By default, malformed episode rows whose declared frame count
    does not match loaded frames raise an error; `skip_malformed_rows=True`
    skips them with a warning and metric.

    For smoke jobs, use `mdr.read_lerobot(path, limit=1)` to emit only the
    first episode before downstream transforms run.
    """
    return RefinerPipeline(
        source=LeRobotEpisodeReader(
            inputs,
            fs=fs,
            storage_options=storage_options,
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            split_row_groups=split_row_groups,
            skip_malformed_rows=skip_malformed_rows,
            limit=limit,
        )
    )


def read_tfrecords(
    inputs: DataFileSetLike,
    *,
    features: Mapping[str, Any],
    fs: AbstractFileSystem | None = None,
    storage_options: Mapping[str, Any] | None = None,
    recursive: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    num_shards: int | None = None,
    batch_size: int = 1024,
    compression: str | None = "auto",
    num_parallel_calls: int | None = None,
    prefetch: int | None = 1,
    file_path_column: str | None = "file_path",
) -> RefinerPipeline:
    """Create a pipeline with a TensorFlow TFRecord reader source.

    Records are parsed with `tf.io.parse_example(features)` and emitted as
    Arrow-backed `Tabular` batches. Sharding is file-level; one large TFRecord
    file is one source shard.

    Args:
        inputs: TFRecord file, glob, directory, or sequence of inputs.
        features: Mapping passed to `tf.io.parse_example`.
        fs: Optional fsspec filesystem for string inputs. TensorFlow reads only
            local resolved paths.
        storage_options: Optional fsspec storage options.
        recursive: Whether directory inputs should be expanded recursively.
        target_shard_bytes: Target shard size used when grouping whole files.
        num_shards: Optional requested number of planned file shards.
        batch_size: Number of serialized examples parsed per batch.
        compression: `None`, `"auto"`, `"gzip"`, or `"zlib"`.
        num_parallel_calls: Optional TensorFlow map parallelism.
        prefetch: Optional TensorFlow prefetch depth.
        file_path_column: Source file path column, or `None` to omit it.
    """
    return RefinerPipeline(
        source=TfrecordReader(
            inputs,
            features=features,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            batch_size=batch_size,
            compression=compression,
            num_parallel_calls=num_parallel_calls,
            prefetch=prefetch,
            file_path_column=file_path_column,
        )
    )


def read_tfds(
    input: str | Sequence[str],
    *,
    config: str | None = None,
    split: str = "train",
    data_dir: str | None = None,
    download: bool = False,
    batch_size: int = 1024,
    examples_per_shard: int = 10_000,
    num_shards: int | None = None,
    shuffle_files: bool = False,
    read_config: Any | None = None,
    decoders: Mapping[str, Any] | None = None,
    as_supervised: bool = False,
    videos: PathSelection | None = None,
    fps: float = 30.0,
) -> RefinerPipeline:
    """Create a pipeline with a TensorFlow Datasets reader source.

    The reader uses TFDS feature decoding and plans row-range shards within a
    single plain split name.

    Args:
        input: TFDS dataset name, prepared TFDS directory, or a sequence of
            prepared directories/names with the same feature schema.
        config: Optional TFDS builder config.
        split: Plain split name from `builder.info.splits`.
        data_dir: Optional local TFDS data directory.
        download: Whether to call `download_and_prepare()`.
        batch_size: Number of decoded examples per emitted batch.
        examples_per_shard: Target examples per shard when `num_shards` is
            omitted.
        num_shards: Optional requested number of row-range shards.
        shuffle_files: Passed to `builder.as_dataset`.
        read_config: Optional TFDS read config.
        decoders: Optional TFDS feature decoders.
        as_supervised: Whether to read supervised `(input, target)` pairs.
        videos: Optional video-name to nested dataset frame path mapping.
        fps: Frame rate used for `videos`.
    """
    return RefinerPipeline(
        source=TfdsReader(
            input,
            config=config,
            split=split,
            data_dir=data_dir,
            download=download,
            batch_size=batch_size,
            examples_per_shard=examples_per_shard,
            num_shards=num_shards,
            shuffle_files=shuffle_files,
            read_config=read_config,
            decoders=decoders,
            as_supervised=as_supervised,
            videos=videos,
            fps=fps,
        )
    )


def from_items(
    items: Sequence[Any],
    *,
    items_per_shard: int = 1_000,
) -> RefinerPipeline:
    """Create a pipeline from in-memory rows.

    Intended for small/medium inline datasets; large datasets should use file-backed
    readers (`read_parquet`/`read_json`/`read_csv`). Primitive items are wrapped
    as ``{"item": value}``.
    """
    return RefinerPipeline(
        source=ItemsSource(
            items=items,
            items_per_shard=items_per_shard,
        )
    )


def from_source(source: BaseSource) -> RefinerPipeline:
    """Create a pipeline from an already-constructed source.

    Use this when implementing or testing custom sources. Most user code should
    prefer a typed reader helper such as ``read_parquet`` or ``read_lerobot``.
    """
    return RefinerPipeline(source=source)


def task(
    fn: Callable[[int, int], Any],
    *,
    num_tasks: int,
) -> RefinerPipeline:
    """Create a task-style pipeline with one callback invocation per rank.

    ``fn`` receives ``(task_rank, num_tasks)`` and is invoked once for each
    integer rank in ``range(num_tasks)``. This is useful for jobs that perform
    side effects or generate work without reading an input dataset.
    """
    source = TaskSource(num_tasks=num_tasks)
    return RefinerPipeline(source=source).add_step(
        TaskStep(fn=fn, num_tasks=num_tasks, index=1)
    )
