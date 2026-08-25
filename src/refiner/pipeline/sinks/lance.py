from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import os
import posixpath
import queue as queue_module
import re
import tempfile
from collections.abc import Sequence
from typing import Any, Literal, get_args

import pyarrow as pa
import pyarrow.compute as pc

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data import datatype
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.shard import INTERNAL_ROW_COLUMNS, SOURCE_ROW_ID_COLUMN
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.assets import (
    AssetUploadManager,
    AssetWriteConfig,
    BlobAssetConfig,
    BlobAssetManager,
    FileAssetConfig,
    asset_config_to_plan,
    cleanup_rejected_asset_attempts,
)
from refiner.pipeline.sinks.lance_schema import (
    lance_schema_from_payload as _lance_schema_from_payload,
    lance_schema_to_payload as _lance_schema_to_payload,
)
from refiner.pipeline.sinks.lance_utils import block_to_table, validate_lance_uri
from refiner.pipeline.sinks.reducer.file import (
    _compile_output_path_patterns,
)
from refiner.utils import check_required_dependencies
from refiner.worker.context import (
    get_active_stage_index,
    get_active_job_id,
    get_active_worker_token,
    get_finalized_workers,
    logger,
)
from refiner.worker.lifecycle import sort_finalized_workers
from refiner.worker.metrics.api import log_throughput

LanceWriteMode = Literal["create", "append", "overwrite", "add_columns"]
_METADATA_FILENAME_TEMPLATE = (
    "_refiner_lance_fragments/{job_id}/{shard_id}__w{worker_id}.jsonl"
)
_QUEUE_CLOSED = object()
_QUEUE_POLL_SECONDS = 0.1
_LANCE_CLEANUP_WORKERS = 16
_LANCE_WRITER_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="refiner-lance-writer",
)
_LANCE_ROW_ADDRESS_FRAGMENT_SHIFT = 32
_LANCE_ROW_ADDRESS_POSITION_MASK = (1 << _LANCE_ROW_ADDRESS_FRAGMENT_SHIFT) - 1


def _import_lance() -> Any:
    check_required_dependencies(
        "write_lance_dataset", [("lance", "pylance")], dist="lance"
    )
    import lance

    return lance


def _schema_difference(expected: pa.Schema, actual: pa.Schema) -> str:
    """Return a compact, actionable schema difference for distributed commits."""
    details: list[str] = []
    expected_names = set(expected.names)
    actual_names = set(actual.names)
    if missing := sorted(expected_names - actual_names):
        details.append("missing=" + ", ".join(missing))
    if extra := sorted(actual_names - expected_names):
        details.append("unexpected=" + ", ".join(extra))
    for name in expected.names:
        actual_index = actual.get_field_index(name)
        if actual_index < 0:
            continue
        expected_field = expected.field(name)
        actual_field = actual.field(actual_index)
        if not expected_field.equals(actual_field, check_metadata=True):
            details.append(
                f"{name}: expected={expected_field!s}, actual={actual_field!s}"
            )
    if expected.metadata != actual.metadata:
        details.append("schema metadata differs")
    return "; ".join(details) or "unknown schema difference"


def _cast_to_planned_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Normalize every worker's materialized output to the planned schema."""
    actual_names = set(table.schema.names)
    expected_names = set(schema.names)
    if actual_names != expected_names:
        raise ValueError(
            "Lance output columns differ from planned schema: "
            + _schema_difference(schema, table.schema)
        )
    # Selecting establishes the planner's deterministic column order; cast uses
    # the target fields/metadata rather than preserving worker-local inference.
    return table.select(schema.names).cast(schema, safe=True)


def _validate_write_mode(mode: str) -> None:
    valid_modes = get_args(LanceWriteMode)
    if mode not in valid_modes:
        raise ValueError("mode must be one of: " + ", ".join(sorted(valid_modes)))


def _asset_value_field(field: pa.Field) -> pa.Field | None:
    if datatype.is_asset_field(field):
        return field
    field_type = field.type
    if (
        pa.types.is_list(field_type)
        or pa.types.is_large_list(field_type)
        or pa.types.is_fixed_size_list(field_type)
    ) and datatype.is_asset_field(field_type.value_field):
        return field_type.value_field
    return None


def _validate_append_asset_layout(
    output_schema: pa.Schema,
    destination_schema: pa.Schema,
) -> None:
    incompatible: list[str] = []
    for output_field in output_schema:
        destination_index = destination_schema.get_field_index(output_field.name)
        if destination_index < 0:
            continue
        destination_field = destination_schema.field(destination_index)
        output_asset = _asset_value_field(output_field)
        destination_asset = _asset_value_field(destination_field)
        if output_asset is None and destination_asset is None:
            continue
        asset_types_match = (
            output_asset is None
            or destination_asset is None
            or datatype.asset_type(output_asset)
            == datatype.asset_type(destination_asset)
        )
        if (
            not output_field.type.equals(destination_field.type)
            or not asset_types_match
        ):
            incompatible.append(output_field.name)
    if incompatible:
        raise ValueError(
            "Cannot append columns with incompatible asset layouts: "
            + ", ".join(sorted(incompatible))
        )


def _schema_to_base64(schema: pa.Schema) -> str:
    return base64.b64encode(schema.serialize().to_pybytes()).decode("ascii")


def _schema_from_base64(value: str) -> pa.Schema:
    return pa.ipc.read_schema(pa.BufferReader(base64.b64decode(value)))


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _job_token(job_id: str) -> str:
    return hashlib.sha256(job_id.encode("utf-8")).hexdigest()


def _metadata_prefix() -> str:
    return f"_refiner_lance_fragments/{_job_token(get_active_job_id())}"


def _finalized_workers(*, reducer_name: str) -> list[Any]:
    stage_index = get_active_stage_index()
    if stage_index is None or stage_index <= 0:
        raise ValueError(
            f"{reducer_name} requires an active reducer stage with a prior writer stage"
        )
    return sort_finalized_workers(get_finalized_workers(stage_index=stage_index - 1))


def _managed_paths(
    *,
    output: DataFolder,
    managed_path_pattern: re.Pattern[str],
    search_path: str,
    reducer_name: str,
    cleanup_path_prefix: str | None = None,
) -> tuple[list[str], list[str]]:
    finalized_workers = _finalized_workers(reducer_name=reducer_name)
    keep_pairs = {(row.shard_id, row.worker_token) for row in finalized_workers}
    predecessor_jobs_by_pair = {
        (row.shard_id, row.worker_token): _job_token(row.job_id)
        for row in finalized_workers
        if row.job_id is not None
    }
    current_paths: set[str] = set()
    if cleanup_path_prefix is not None:
        try:
            listed = output.find(cleanup_path_prefix)
        except FileNotFoundError:
            listed = []
        current_paths = {
            rel_path
            for rel_path in listed
            if isinstance(rel_path, str)
            and managed_path_pattern.fullmatch(rel_path) is not None
        }

    finalized_by_pair: dict[tuple[str, str], str] = {}
    for rel_path in current_paths:
        match = managed_path_pattern.fullmatch(rel_path)
        assert match is not None
        pair = (match.group("shard_id"), match.group("worker_id"))
        if pair in keep_pairs:
            finalized_by_pair[pair] = rel_path

    missing_pairs = keep_pairs.difference(finalized_by_pair)
    historical_paths: set[str] = set()
    if missing_pairs:
        listed: list[str] = []
        known_jobs = {
            predecessor_jobs_by_pair[pair]
            for pair in missing_pairs
            if pair in predecessor_jobs_by_pair
        }
        for job_id in sorted(known_jobs):
            try:
                listed.extend(output.find(f"{search_path}/{job_id}"))
            except FileNotFoundError:
                continue
        if any(pair not in predecessor_jobs_by_pair for pair in missing_pairs):
            try:
                listed.extend(output.find(search_path))
            except FileNotFoundError:
                pass
        candidates: dict[tuple[str, str], set[str]] = {
            pair: set() for pair in missing_pairs
        }
        for rel_path in listed:
            if not isinstance(rel_path, str) or rel_path in current_paths:
                continue
            match = managed_path_pattern.fullmatch(rel_path)
            if match is None:
                continue
            pair = (match.group("shard_id"), match.group("worker_id"))
            predecessor_job = predecessor_jobs_by_pair.get(pair)
            if pair in candidates and (
                predecessor_job is None
                or posixpath.dirname(rel_path) == f"{search_path}/{predecessor_job}"
            ):
                candidates[pair].add(rel_path)

        selected_prefixes: set[str] = set()
        for pair, paths in candidates.items():
            if len(paths) > 1:
                raise ValueError(
                    "Ambiguous resumed Lance metadata for shard/worker "
                    f"{pair[0]}/{pair[1]}"
                )
            if paths:
                selected = next(iter(paths))
                finalized_by_pair[pair] = selected
                selected_prefixes.add(posixpath.dirname(selected))
        historical_paths = {
            rel_path
            for rel_path in listed
            if isinstance(rel_path, str)
            and posixpath.dirname(rel_path) in selected_prefixes
            and managed_path_pattern.fullmatch(rel_path) is not None
        }

    missing_pairs = keep_pairs.difference(finalized_by_pair)
    if missing_pairs:
        missing = ", ".join(
            f"{shard_id}/{worker_id}" for shard_id, worker_id in sorted(missing_pairs)
        )
        raise ValueError(f"Missing Lance metadata for finalized workers: {missing}")

    selected_paths = [
        finalized_by_pair[(row.shard_id, row.worker_token)]
        for row in finalized_workers
        if (row.shard_id, row.worker_token) in finalized_by_pair
    ]
    return selected_paths, sorted(current_paths | historical_paths)


def _fragment_data_paths(fragment_json: str) -> list[str]:
    payload = json.loads(fragment_json)
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        return []

    paths: list[str] = []
    for file_info in files:
        path = file_info.get("path") if isinstance(file_info, dict) else None
        if not isinstance(path, str):
            continue
        normalized = posixpath.normpath(path)
        if (
            normalized.startswith("../")
            or normalized == ".."
            or normalized.startswith("/")
        ):
            raise ValueError(f"Invalid Lance fragment file path: {path}")
        paths.append(posixpath.join("data", normalized))
    return paths


def _validate_created_file_path(path: object) -> str:
    if not isinstance(path, str):
        raise ValueError("Invalid Lance created-file path")
    normalized = posixpath.normpath(path)
    if (
        normalized != path
        or normalized == "data"
        or not normalized.startswith("data/")
        or normalized.startswith("/")
        or ".." in normalized.split("/")
    ):
        raise ValueError(f"Invalid Lance created-file path: {path}")
    return normalized


def _remove_fragment_data(output: DataFolder, fragment_json: str) -> None:
    for rel_path in _fragment_data_paths(fragment_json):
        try:
            output.rm(rel_path)
        except FileNotFoundError:
            continue


def _remove_paths_best_effort(
    output: DataFolder,
    paths: Sequence[str],
    *,
    operation: str,
) -> None:
    unique_paths = tuple(dict.fromkeys(paths))
    if not unique_paths:
        return

    def _remove(path: str) -> None:
        try:
            output.rm(path)
        except FileNotFoundError:
            return
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "{} failed path={}: {}: {}",
                operation,
                path,
                type(err).__name__,
                err,
            )

    max_workers = min(_LANCE_CLEANUP_WORKERS, len(unique_paths))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_remove, unique_paths))


class _StreamingShardWriter:
    def __init__(
        self,
        *,
        dataset_uri: str,
        schema: pa.Schema,
        mode: LanceWriteMode,
    ) -> None:
        self.dataset_uri = dataset_uri
        self.schema = schema
        self.mode = mode
        self.queue: queue_module.Queue[pa.RecordBatch | object] = queue_module.Queue(
            maxsize=8
        )
        self.closed = False
        self.task_future = _LANCE_WRITER_POOL.submit(self._run)
        self._spool_path: str | None = None
        self._spool_sink: pa.OSFile | None = None
        self._spool_writer: pa.RecordBatchStreamWriter | None = None
        if not self.task_future.running() and self.task_future.cancel():
            tmp = tempfile.NamedTemporaryFile(prefix="refiner-lance-", delete=False)
            self._spool_path = tmp.name
            tmp.close()
            self._spool_sink = pa.OSFile(self._spool_path, "wb")
            self._spool_writer = pa.ipc.new_stream(self._spool_sink, self.schema)

    def _iter_batches(self):
        while True:
            item = self.queue.get()
            if item is _QUEUE_CLOSED:
                return
            yield item

    def _write_reader(self, reader: pa.RecordBatchReader) -> list[str]:
        lance = _import_lance()
        fragments = lance.fragment.write_fragments(
            reader,
            self.dataset_uri,
            schema=self.schema,
            mode=self.mode,
        )
        return [_json_dumps(fragment.to_json()) for fragment in fragments]

    def _run(self) -> list[str]:
        return self._write_reader(
            pa.RecordBatchReader.from_batches(self.schema, self._iter_batches())
        )

    def _raise_if_failed(self) -> None:
        if self._spool_writer is not None:
            return
        if not self.task_future.done():
            return
        error = self.task_future.exception()
        if error is not None:
            raise RuntimeError("Lance fragment writer failed") from error

    def _put(self, item: pa.RecordBatch | object) -> None:
        while True:
            self._raise_if_failed()
            try:
                self.queue.put(item, timeout=_QUEUE_POLL_SECONDS)
                return
            except queue_module.Full:
                continue

    def put_batches(self, batches: list[pa.RecordBatch]) -> None:
        if self.closed:
            raise RuntimeError("Cannot write to a closed Lance shard writer.")
        for batch in batches:
            if self._spool_writer is not None:
                self._spool_writer.write_batch(batch)
            else:
                self._put(batch)

    def finish(self) -> list[str]:
        if not self.closed:
            self.closed = True
            if self._spool_writer is None:
                self._put(_QUEUE_CLOSED)
        if self._spool_writer is not None:
            assert self._spool_sink is not None
            assert self._spool_path is not None
            self._spool_writer.close()
            self._spool_sink.close()
            self._spool_writer = None
            try:
                with pa.memory_map(self._spool_path, "r") as source:
                    return self._write_reader(pa.ipc.open_stream(source))
            finally:
                os.unlink(self._spool_path)
        return list(self.task_future.result())


class _StreamingAddColumnsWriter:
    def __init__(
        self,
        *,
        fragment: Any,
        fragment_id: int,
        num_rows: int,
        schema: pa.Schema,
        output: DataFolder,
    ) -> None:
        self.fragment = fragment
        self.fragment_id = fragment_id
        self.num_rows = num_rows
        self.schema = schema
        self.output = output
        self.base_json = _json_dumps(fragment.metadata.to_json())
        self.queue: queue_module.Queue[pa.RecordBatch | object] = queue_module.Queue(
            maxsize=8
        )
        self.closed = False
        self.next_position = 0
        self.pending: dict[int, tuple[int, pa.Table]] = {}
        self.task_future = _LANCE_WRITER_POOL.submit(self._run)
        self._spool_path: str | None = None
        self._spool_sink: pa.OSFile | None = None
        self._spool_writer: pa.RecordBatchStreamWriter | None = None
        if not self.task_future.running() and self.task_future.cancel():
            tmp = tempfile.NamedTemporaryFile(
                prefix="refiner-lance-columns-", delete=False
            )
            self._spool_path = tmp.name
            tmp.close()
            self._spool_sink = pa.OSFile(self._spool_path, "wb")
            self._spool_writer = pa.ipc.new_stream(self._spool_sink, self.schema)

    def _iter_batches(self):
        while True:
            item = self.queue.get()
            if item is _QUEUE_CLOSED:
                return
            yield item

    def _merge_reader(self, reader: pa.RecordBatchReader) -> tuple[Any, Any]:
        return self.fragment.merge_columns(reader, reader_schema=self.schema)

    def _run(self) -> tuple[Any, Any]:
        return self._merge_reader(
            pa.RecordBatchReader.from_batches(self.schema, self._iter_batches())
        )

    def _raise_if_failed(self) -> None:
        if self._spool_writer is not None or not self.task_future.done():
            return
        error = self.task_future.exception()
        if error is not None:
            raise RuntimeError("Lance add-columns writer failed") from error

    def _put_batch(self, batch: pa.RecordBatch) -> None:
        if self._spool_writer is not None:
            self._spool_writer.write_batch(batch)
            return
        while True:
            self._raise_if_failed()
            try:
                self.queue.put(batch, timeout=_QUEUE_POLL_SECONDS)
                return
            except queue_module.Full:
                continue

    def _emit_ready(self) -> None:
        while self.next_position in self.pending:
            end, table = self.pending.pop(self.next_position)
            for batch in table.to_batches():
                self._put_batch(batch)
            self.next_position = end

    def put(self, positions: pa.ChunkedArray, output: pa.Table) -> None:
        if self.closed:
            raise RuntimeError("Cannot write to a closed Lance add-columns writer.")
        positions = pc.cast(positions, pa.uint64()).combine_chunks()
        if positions.null_count:
            raise ValueError("Lance row position cannot be null")
        indices = pc.call_function("sort_indices", [positions])
        sorted_positions = pc.take(positions, indices)
        sorted_output = output.take(indices)
        position_values = sorted_positions.to_pylist()
        if not position_values:
            return

        run_start = 0
        for index in range(1, len(position_values) + 1):
            if (
                index < len(position_values)
                and position_values[index] == position_values[index - 1] + 1
            ):
                continue
            start = int(position_values[run_start])
            end = int(position_values[index - 1]) + 1
            if start < self.next_position or end > self.num_rows:
                raise ValueError(
                    f"Lance fragment {self.fragment_id} has invalid or duplicate "
                    "row positions"
                )
            if any(
                start < pending_end and end > pending_start
                for pending_start, (pending_end, _) in self.pending.items()
            ):
                raise ValueError(
                    f"Lance fragment {self.fragment_id} has invalid or duplicate "
                    "row positions"
                )
            self.pending[start] = (
                end,
                sorted_output.slice(run_start, index - run_start),
            )
            run_start = index
        self._emit_ready()

    def _close_input(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._spool_writer is not None:
            assert self._spool_sink is not None
            self._spool_writer.close()
            self._spool_sink.close()
            self._spool_writer = None
        else:
            while True:
                self._raise_if_failed()
                try:
                    self.queue.put(_QUEUE_CLOSED, timeout=_QUEUE_POLL_SECONDS)
                    break
                except queue_module.Full:
                    continue

    def _cleanup_result(self, result: tuple[Any, Any]) -> None:
        updated_fragment, _ = result
        updated_json = _json_dumps(updated_fragment.to_json())
        created_files = sorted(
            set(_fragment_data_paths(updated_json)).difference(
                _fragment_data_paths(self.base_json)
            )
        )
        _remove_paths_best_effort(
            self.output,
            created_files,
            operation="partial Lance add-columns cleanup",
        )

    def finish(self) -> tuple[Any, Any]:
        complete = self.next_position == self.num_rows and not self.pending
        self._close_input()
        if not complete:
            if self._spool_path is None:
                try:
                    result = self.task_future.result()
                except Exception:  # noqa: BLE001
                    pass
                else:
                    self._cleanup_result(result)
            else:
                os.unlink(self._spool_path)
            raise ValueError(
                f"Lance fragment {self.fragment_id} has missing row positions"
            )
        if self._spool_path is not None:
            try:
                with pa.memory_map(self._spool_path, "r") as source:
                    return self._merge_reader(pa.ipc.open_stream(source))
            finally:
                os.unlink(self._spool_path)
        return self.task_future.result()

    def abort(self) -> None:
        self._close_input()
        if self._spool_path is not None:
            os.unlink(self._spool_path)
            return
        try:
            result = self.task_future.result()
        except Exception:  # noqa: BLE001
            pass
        else:
            self._cleanup_result(result)


class LanceDatasetSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        mode: LanceWriteMode = "create",
        columns: Sequence[str] | None = None,
        source_uri: str | None = None,
        source_version: int | None = None,
        assets: AssetWriteConfig | None = None,
    ) -> None:
        _validate_write_mode(mode)
        if mode == "add_columns" and not columns:
            raise ValueError("add_columns requires at least one output column")
        if columns is not None and len(set(columns)) != len(columns):
            raise ValueError("Lance output columns must be unique")
        if columns is not None:
            reserved_columns = set(INTERNAL_ROW_COLUMNS).intersection(columns)
            if reserved_columns:
                raise ValueError(f"{sorted(reserved_columns)[0]} is an internal column")
        if mode != "add_columns" and columns is not None:
            raise ValueError("columns is only supported with mode='add_columns'")
        if mode == "add_columns" and (source_uri is None or source_version is None):
            raise ValueError("add_columns requires a version-pinned Lance source")
        if (
            mode == "add_columns"
            and assets is not None
            and assets.missing_policy == "drop_row"
        ):
            raise ValueError(
                "add_columns cannot use assets with missing_policy='drop_row'"
            )
        self.output = DataFolder.resolve(output)
        if self.output.has_explicit_filesystem_configuration:
            raise ValueError(
                "write_lance_dataset does not support configured fsspec handles; "
                "pass a URI whose credentials and endpoint are available to Lance"
            )
        validate_lance_uri(self.output.abs_path())
        if source_uri is not None:
            validate_lance_uri(source_uri)
        self.mode = mode
        self.columns = tuple(columns) if columns is not None else None
        self.source_uri = source_uri
        self.source_version = source_version
        self.assets = assets
        if mode == "add_columns" and self.output.abs_path() != source_uri:
            raise ValueError("add_columns must write back to the loaded Lance dataset")
        if isinstance(assets, FileAssetConfig):
            self._assets = AssetUploadManager(
                self.output,
                assets_subdir=assets.subdir,
                filename_template=_METADATA_FILENAME_TEMPLATE,
                max_uploads_in_flight=assets.max_in_flight,
                missing_asset_policy=assets.missing_policy,
            )
        elif isinstance(assets, BlobAssetConfig):
            self._assets = BlobAssetManager(
                self.output,
                config=assets,
                filename_template=_METADATA_FILENAME_TEMPLATE,
            )
        else:
            self._assets = None
        self._writers_by_shard: dict[str, _StreamingShardWriter] = {}
        self._schema_by_shard: dict[str, pa.Schema] = {}
        self._add_columns_writers_by_shard: dict[
            str, dict[int, _StreamingAddColumnsWriter]
        ] = {}
        self._add_columns_schema: pa.Schema | None = None
        self._planned_output_schema: pa.Schema | None = None
        self._existing_schema: pa.Schema | None = None
        self._existing_version: int | None = None
        self._source_dataset_cache: Any | None = None

    def _declared_refiner_extras(self) -> tuple[str, ...]:
        return ("lance",)

    def _dataset_uri(self) -> str:
        return self.output.abs_path()

    def _source_dataset(self) -> Any:
        assert self.source_uri is not None
        assert self.source_version is not None
        if self._source_dataset_cache is None:
            self._source_dataset_cache = _import_lance().dataset(
                self.source_uri, version=self.source_version
            )
        return self._source_dataset_cache

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_source_dataset_cache"] = None
        return state

    def _relpath(self, shard_id: str) -> str:
        return _METADATA_FILENAME_TEMPLATE.format(
            job_id=_job_token(get_active_job_id()),
            shard_id=shard_id,
            worker_id=get_active_worker_token(),
        )

    def _load_existing_schema(self) -> pa.Schema:
        if self._existing_schema is not None:
            return self._existing_schema
        lance = _import_lance()
        try:
            dataset = lance.dataset(self._dataset_uri())
        except (FileNotFoundError, OSError, ValueError) as err:
            message = str(err).lower()
            if "not found" in message or "does not exist" in message:
                raise ValueError(
                    "Cannot append to a non-existent Lance dataset."
                ) from err
            raise
        self._existing_schema = dataset.schema
        self._existing_version = int(dataset.version)
        return self._existing_schema

    def _load_overwrite_version(self) -> int:
        if self._existing_version is not None:
            return self._existing_version
        try:
            dataset = _import_lance().dataset(self._dataset_uri())
        except (FileNotFoundError, OSError, ValueError) as err:
            message = str(err).lower()
            if "not found" not in message and "does not exist" not in message:
                raise
            self._existing_version = 0
        else:
            self._existing_version = int(dataset.version)
        return self._existing_version

    def set_input_schema(self, schema: pa.Schema | None) -> None:
        if self._assets is not None:
            self._assets.set_input_schema(schema)
            schema = self._assets.output_schema(schema)
        # The planner has a complete schema for Lance-backed sources and for
        # maps with dtypes.  Keep it so independent cloud workers cannot leak
        # local Arrow inference or metadata into their output fragments.
        self._planned_output_schema = schema
        if self.mode != "add_columns":
            return
        assert self.columns is not None
        assert self.source_uri is not None
        assert self.source_version is not None
        source_schema = self._source_dataset().schema
        conflicts = sorted(set(self.columns).intersection(source_schema.names))
        if conflicts:
            raise ValueError(
                "add_columns cannot replace existing columns: " + ", ".join(conflicts)
            )
        if schema is None:
            return
        missing = sorted(set(self.columns).difference(schema.names))
        if not missing:
            self._add_columns_schema = pa.schema(
                [schema.field(column) for column in self.columns]
            )

    def _write_add_columns_block(self, shard_id: str, block: Block) -> None:
        if not isinstance(block, Tabular) and not block:
            return
        tabular = (
            block
            if isinstance(block, Tabular)
            else block[0].tabular_type.from_rows(block)
        )
        if SOURCE_ROW_ID_COLUMN not in tabular.table.column_names:
            raise ValueError("sink input is missing source row identities")
        row_addresses = tabular.table.column(SOURCE_ROW_ID_COLUMN).combine_chunks()
        if row_addresses.null_count:
            raise ValueError("sink input source row identities cannot contain nulls")
        if row_addresses.type != pa.uint64():
            raise ValueError("sink input source row identities must be uint64")
        table = block_to_table(tabular)
        if table.num_rows == 0:
            return
        assert self.columns is not None
        missing = sorted(set(self.columns).difference(table.schema.names))
        if missing:
            raise ValueError(
                "add_columns output is missing columns: " + ", ".join(missing)
            )
        table = table.select(self.columns)
        if self._assets is not None:
            table = self._assets.rewrite_table(shard_id, table)

        output_schema = pa.schema(
            [table.schema.field(column) for column in self.columns]
        )
        if self._add_columns_schema is None:
            self._add_columns_schema = output_schema
        elif not self._add_columns_schema.equals(output_schema, check_metadata=True):
            raise ValueError("add_columns output schema changed between blocks")
        fragment_ids = pc.call_function(
            "shift_right",
            [row_addresses, pa.scalar(_LANCE_ROW_ADDRESS_FRAGMENT_SHIFT, pa.uint64())],
        )
        writers = self._add_columns_writers_by_shard.setdefault(shard_id, {})
        unique_fragment_ids = pc.call_function("unique", [fragment_ids])
        for fragment_id_raw in unique_fragment_ids.to_pylist():
            fragment_id = int(fragment_id_raw)
            mask = pc.call_function(
                "equal",
                [fragment_ids, pa.scalar(fragment_id, type=pa.uint64())],
            )
            writer = writers.get(fragment_id)
            if writer is None:
                fragment = self._source_dataset().get_fragment(fragment_id)
                if int(fragment.num_deletions) > 0:
                    raise ValueError(
                        f"Lance fragment {fragment_id} has deletions; add_columns "
                        "does not yet support deletion-bearing fragments"
                    )
                writer = _StreamingAddColumnsWriter(
                    fragment=fragment,
                    fragment_id=fragment_id,
                    num_rows=int(fragment.physical_rows),
                    schema=output_schema,
                    output=self.output,
                )
                writers[fragment_id] = writer
            positions = pc.call_function(
                "bit_wise_and",
                [
                    pc.call_function("filter", [row_addresses, mask]),
                    pa.scalar(_LANCE_ROW_ADDRESS_POSITION_MASK, pa.uint64()),
                ],
            )
            writer.put(
                pa.chunked_array([positions]),
                table.filter(mask).select(self.columns),
            )

    def write_shard_block(self, shard_id: str, block: Block) -> None:
        if self.mode == "add_columns":
            self._write_add_columns_block(shard_id, block)
            return
        table = block_to_table(block)
        if table.num_rows == 0:
            return
        if self._assets is not None:
            table = self._assets.rewrite_table(shard_id, table)
        if self._planned_output_schema is not None:
            table = _cast_to_planned_schema(table, self._planned_output_schema)
        if self.mode == "append":
            existing_schema = self._load_existing_schema()
            _validate_append_asset_layout(table.schema, existing_schema)
            table = table.cast(existing_schema)
        elif self.mode == "overwrite":
            self._load_overwrite_version()

        existing_schema = self._schema_by_shard.setdefault(shard_id, table.schema)
        if not existing_schema.equals(table.schema, check_metadata=True):
            raise ValueError("Cannot write one Lance shard with inconsistent schemas.")
        writer = self._writers_by_shard.get(shard_id)
        if writer is None:
            writer = _StreamingShardWriter(
                dataset_uri=self._dataset_uri(),
                schema=table.schema,
                mode=self.mode,
            )
            self._writers_by_shard[shard_id] = writer
        writer.put_batches(table.to_batches())

    def _write_sidecar(
        self,
        shard_id: str,
        payload: dict[str, object],
        *,
        created_files: Sequence[str] = (),
    ) -> None:
        try:
            with self.output.open(
                self._relpath(shard_id), mode="wt", encoding="utf-8"
            ) as f:
                f.write(_json_dumps(payload))
                f.write("\n")
        except Exception:
            _remove_paths_best_effort(
                self.output,
                created_files,
                operation="failed Lance sidecar cleanup",
            )
            raise
        log_throughput("files_written", 1, shard_id=shard_id, unit="files")

    def _write_empty_sidecar(self, shard_id: str) -> None:
        payload: dict[str, object] = {
            "empty": True,
            "fragments": [],
            "created_files": [],
        }
        if self.mode == "append":
            self._load_existing_schema()
        elif self.mode == "overwrite":
            self._load_overwrite_version()
        if self.mode in ("append", "overwrite"):
            payload["source_version"] = self._existing_version
        elif self.mode == "add_columns":
            payload["source_version"] = self.source_version
        self._write_sidecar(shard_id, payload)

    def on_shard_complete(self, shard_id: str) -> None:
        if self._assets is not None:
            self._assets.on_shard_complete(shard_id)
        if self.mode == "add_columns":
            self._complete_add_columns_shard(shard_id)
            return
        writer = self._writers_by_shard.pop(shard_id, None)
        schema = self._schema_by_shard.pop(shard_id, None)
        if writer is None or schema is None:
            self._write_empty_sidecar(shard_id)
            return
        fragments = writer.finish()
        if not fragments:
            self._write_empty_sidecar(shard_id)
            return
        created_files = [
            path for fragment in fragments for path in _fragment_data_paths(fragment)
        ]
        payload: dict[str, object] = {
            "schema": _schema_to_base64(schema),
            "fragments": fragments,
            "created_files": sorted(created_files),
        }
        if self.mode in ("append", "overwrite"):
            payload["source_version"] = self._existing_version
        self._write_sidecar(
            shard_id,
            payload,
            created_files=created_files,
        )

    def _complete_add_columns_shard(self, shard_id: str) -> None:
        writers = self._add_columns_writers_by_shard.pop(shard_id, None)
        if not writers:
            self._write_empty_sidecar(shard_id)
            return
        assert self.source_version is not None
        updated_jsons: list[str] = []
        created_files: list[str] = []
        merged_schema: Any | None = None
        lance_schema_payload: dict[str, object] | None = None
        try:
            for fragment_id, writer in sorted(writers.items()):
                updated_fragment, next_schema = writer.finish()
                updated_json = _json_dumps(updated_fragment.to_json())
                updated_jsons.append(updated_json)
                created_files.extend(
                    set(_fragment_data_paths(updated_json)).difference(
                        _fragment_data_paths(writer.base_json)
                    )
                )
                next_payload = _lance_schema_to_payload(next_schema)
                if merged_schema is None:
                    merged_schema = next_schema
                    lance_schema_payload = next_payload
                elif merged_schema != next_schema:
                    raise ValueError(
                        "Cannot write one Lance shard with inconsistent field IDs."
                    )
        except Exception:
            for writer in writers.values():
                try:
                    writer.abort()
                except Exception:  # noqa: BLE001
                    pass
            _remove_paths_best_effort(
                self.output,
                created_files,
                operation="failed Lance shard cleanup",
            )
            raise
        assert merged_schema is not None
        assert lance_schema_payload is not None
        payload = {
            "schema": _schema_to_base64(merged_schema.to_pyarrow()),
            "lance_schema": lance_schema_payload,
            "fragments": updated_jsons,
            "created_files": sorted(created_files),
            "source_version": self.source_version,
            "source_fragment_ids": sorted(writers),
        }
        self._write_sidecar(
            shard_id,
            payload,
            created_files=created_files,
        )

    def close(self) -> None:
        first_error: Exception | None = None
        if self._assets is not None:
            self._assets.close()
        for writer in self._writers_by_shard.values():
            try:
                for fragment in writer.finish():
                    _remove_fragment_data(self.output, fragment)
            except Exception as err:  # noqa: BLE001
                if first_error is None:
                    first_error = err
        self._writers_by_shard.clear()
        self._schema_by_shard.clear()
        for writers in self._add_columns_writers_by_shard.values():
            for writer in writers.values():
                try:
                    writer.abort()
                except Exception as err:  # noqa: BLE001
                    if first_error is None:
                        first_error = err
        self._add_columns_writers_by_shard.clear()
        if first_error is not None:
            raise first_error

    def describe(self) -> tuple[str, str, dict[str, object]]:
        args: dict[str, object] = {
            "path": self.output.abs_path(),
            "mode": self.mode,
        }
        if self.columns is not None:
            args["columns"] = list(self.columns)
        if self.source_version is not None:
            args["source_version"] = self.source_version
        if self.assets is not None:
            args["assets"] = asset_config_to_plan(self.assets)
        return ("write_lance_dataset", "writer", args)

    def build_reducer(self) -> BaseSink | None:
        return LanceDatasetCommitReducerSink(
            self.output,
            mode=self.mode,
            source_version=self.source_version,
            assets_subdir=self.assets.subdir if self.assets is not None else None,
        )


class LanceDatasetCommitReducerSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        mode: LanceWriteMode,
        source_version: int | None = None,
        assets_subdir: str | None = None,
    ) -> None:
        _validate_write_mode(mode)
        self.output = DataFolder.resolve(output)
        validate_lance_uri(self.output.abs_path())
        self.mode = mode
        self.source_version = source_version
        self.assets_subdir = assets_subdir
        self._managed_path_pattern = _compile_output_path_patterns(
            _METADATA_FILENAME_TEMPLATE
        )[-1]
        self._commit_ran = False
        self._pending_metadata_cleanup: tuple[str, ...] = ()

    def _declared_refiner_extras(self) -> tuple[str, ...]:
        return ("lance",)

    def _dataset_uri(self) -> str:
        return self.output.abs_path()

    @property
    def counts_output_rows(self) -> bool:
        return False

    def describe(self) -> tuple[str, str, dict[str, object]]:
        args: dict[str, object] = {
            "path": self.output.abs_path(),
            "mode": self.mode,
        }
        if self.source_version is not None:
            args["source_version"] = self.source_version
        if self.assets_subdir is not None:
            args["assets_subdir"] = self.assets_subdir
        return ("write_lance_dataset_commit", "writer", args)

    def write_shard_block(self, shard_id: str, block: Block) -> None:
        del shard_id, block
        self._run_commit()

    def on_shard_finalized(self, shard_id: str) -> None:
        del shard_id
        _remove_paths_best_effort(
            self.output,
            self._pending_metadata_cleanup,
            operation="post-commit Lance metadata cleanup",
        )
        self._pending_metadata_cleanup = ()

    def _commit_message(
        self,
        *,
        schema: pa.Schema,
        fragments: Sequence[str],
        source_versions: set[int],
        lance_schema_payload: dict[str, object] | None,
    ) -> str:
        payload = {
            "mode": self.mode,
            "schema": _schema_to_base64(schema),
            "fragments": sorted(fragments),
            "source_versions": sorted(source_versions),
            "lance_schema": lance_schema_payload,
        }
        digest = hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()
        return f"refiner:{digest}"

    def _was_committed(
        self,
        lance: Any,
        commit_message: str,
        *,
        expected_versions: Sequence[int] = (),
        search_history: bool = False,
    ) -> bool:
        existing = self._load_existing_dataset(lance)
        if existing is None:
            return False
        candidate_versions = {int(existing.version), *expected_versions}
        if search_history:
            candidate_versions = {
                int(version_info["version"])
                for version_info in existing.versions()
                if isinstance(version_info.get("version"), int)
            }
        for version in sorted(candidate_versions, reverse=True):
            try:
                transaction = existing.read_transaction(version)
            except (FileNotFoundError, OSError, ValueError):
                continue
            properties = getattr(transaction, "transaction_properties", None)
            if (
                isinstance(properties, dict)
                and properties.get("__lance_commit_message") == commit_message
            ):
                return True
        return False

    def _read_metadata(
        self, rel_path: str
    ) -> tuple[
        pa.Schema | None,
        list[str],
        list[str],
        int | None,
        list[int],
        dict[str, object] | None,
    ]:
        with self.output.open(rel_path, mode="rt", encoding="utf-8") as f:
            payload = json.load(f)
        empty_raw = payload.get("empty", False) if isinstance(payload, dict) else None
        schema_raw = payload.get("schema") if isinstance(payload, dict) else None
        fragment_raw = payload.get("fragments") if isinstance(payload, dict) else None
        created_raw = (
            payload.get("created_files", []) if isinstance(payload, dict) else []
        )
        source_version_raw = (
            payload.get("source_version") if isinstance(payload, dict) else None
        )
        source_fragments_raw = (
            payload.get("source_fragment_ids") if isinstance(payload, dict) else None
        )
        if source_fragments_raw is None and isinstance(payload, dict):
            source_fragment_raw = payload.get("source_fragment_id")
            source_fragments_raw = (
                [] if source_fragment_raw is None else [source_fragment_raw]
            )
        lance_schema_raw = (
            payload.get("lance_schema") if isinstance(payload, dict) else None
        )
        if (
            not isinstance(empty_raw, bool)
            or not isinstance(fragment_raw, list)
            or not all(isinstance(fragment, str) for fragment in fragment_raw)
        ):
            raise ValueError(f"Invalid Lance metadata payload: {rel_path}")
        if not isinstance(created_raw, list):
            raise ValueError(f"Invalid Lance created-files payload: {rel_path}")
        if not isinstance(source_fragments_raw, list) or not all(
            isinstance(fragment_id, int) and not isinstance(fragment_id, bool)
            for fragment_id in source_fragments_raw
        ):
            raise ValueError(f"Invalid Lance source-fragments payload: {rel_path}")
        fragments = [fragment for fragment in fragment_raw if isinstance(fragment, str)]
        created_files = [_validate_created_file_path(path) for path in created_raw]
        if empty_raw:
            if fragments or created_files or source_fragments_raw:
                raise ValueError(f"Invalid empty Lance metadata payload: {rel_path}")
        elif not isinstance(schema_raw, str):
            raise ValueError(f"Invalid Lance metadata payload: {rel_path}")
        return (
            _schema_from_base64(schema_raw) if isinstance(schema_raw, str) else None,
            fragments,
            created_files,
            int(source_version_raw) if source_version_raw is not None else None,
            [int(fragment_id) for fragment_id in source_fragments_raw],
            lance_schema_raw if isinstance(lance_schema_raw, dict) else None,
        )

    def _load_existing_dataset(self, lance: Any) -> Any | None:
        try:
            return lance.dataset(self._dataset_uri())
        except (FileNotFoundError, OSError, ValueError) as err:
            message = str(err).lower()
            if "not found" in message or "does not exist" in message:
                return None
            raise

    def _verified_created_files(
        self,
        lance: Any,
        *,
        fragments: Sequence[str],
        created_files: Sequence[str],
        source_version: int | None,
        source_fragment_ids: Sequence[int],
        metadata_path: str,
    ) -> list[str]:
        if self._managed_path_pattern.fullmatch(metadata_path) is None:
            raise ValueError(f"Invalid Lance metadata path: {metadata_path}")
        if not fragments and not created_files and not source_fragment_ids:
            if self.mode == "add_columns" and source_version != self.source_version:
                raise ValueError("Invalid empty Lance add-columns metadata")
            return []
        if self.mode != "add_columns":
            expected = {
                path
                for fragment in fragments
                for path in _fragment_data_paths(fragment)
            }
            if set(created_files) != expected:
                raise ValueError(
                    "Lance created-files metadata does not match fragment data"
                )
            return list(created_files)
        if (
            source_version != self.source_version
            or not source_fragment_ids
            or len(fragments) != len(source_fragment_ids)
            or len(set(source_fragment_ids)) != len(source_fragment_ids)
        ):
            raise ValueError("Invalid rejected Lance add-columns metadata")
        source = lance.dataset(self._dataset_uri(), version=source_version)
        expected: set[str] = set()
        for fragment, source_fragment_id in zip(
            fragments, source_fragment_ids, strict=True
        ):
            base_fragment = source.get_fragment(source_fragment_id)
            base_json = _json_dumps(base_fragment.metadata.to_json())
            expected.update(
                set(_fragment_data_paths(fragment)).difference(
                    _fragment_data_paths(base_json)
                )
            )
        if set(created_files) != expected:
            raise ValueError(
                "Lance created-files metadata does not match fragment data"
            )
        return list(created_files)

    def _validate_add_columns_fragment_coverage(
        self,
        lance: Any,
        source_fragment_ids: set[int],
    ) -> None:
        if self.mode != "add_columns":
            return
        if self.source_version is None:
            raise ValueError("add_columns reducer is missing its source version")
        source = lance.dataset(
            self._dataset_uri(),
            version=self.source_version,
        )
        expected_fragment_ids = {
            int(fragment.fragment_id)
            for fragment in source.get_fragments()
            if int(fragment.count_rows()) > 0
        }
        missing = sorted(expected_fragment_ids.difference(source_fragment_ids))
        unexpected = sorted(source_fragment_ids.difference(expected_fragment_ids))
        if missing:
            raise ValueError(
                "Missing Lance fragment results: "
                + ", ".join(str(fragment_id) for fragment_id in missing)
            )
        if unexpected:
            raise ValueError(
                "Unexpected Lance fragment results: "
                + ", ".join(str(fragment_id) for fragment_id in unexpected)
            )

    def _commit_empty_output(self, lance: Any) -> None:
        existing = self._load_existing_dataset(lance)
        if self.mode == "append":
            if existing is None:
                raise ValueError("Cannot append to a non-existent Lance dataset.")
            return
        if self.mode == "add_columns":
            self._validate_add_columns_fragment_coverage(lance, set())
            raise ValueError("Cannot add columns to an empty Lance dataset")
        raise ValueError(f"Cannot {self.mode} an empty Lance dataset")

    def _run_commit(self) -> None:
        if self._commit_ran:
            return
        self._commit_ran = True

        metadata_paths, cleanup_paths = _managed_paths(
            output=self.output,
            managed_path_pattern=self._managed_path_pattern,
            search_path="_refiner_lance_fragments",
            reducer_name="write_lance_dataset_commit",
            cleanup_path_prefix=_metadata_prefix(),
        )
        if self.assets_subdir is not None:
            finalized = _finalized_workers(reducer_name="write_lance_dataset_commit")
            cleanup_rejected_asset_attempts(
                self.output,
                self.assets_subdir,
                {(row.shard_id, row.worker_token) for row in finalized},
            )
        rejected_paths = sorted(set(cleanup_paths).difference(metadata_paths))

        rejected_created_files: list[str] = []
        cleanup_lance = _import_lance() if self.mode == "add_columns" else None
        for rel_path in rejected_paths:
            try:
                (
                    _,
                    next_rejected_fragments,
                    next_created_files,
                    next_source_version,
                    next_source_fragment_ids,
                    _,
                ) = self._read_metadata(rel_path)
                next_created_files = self._verified_created_files(
                    cleanup_lance,
                    fragments=next_rejected_fragments,
                    created_files=next_created_files,
                    source_version=next_source_version,
                    source_fragment_ids=next_source_fragment_ids,
                    metadata_path=rel_path,
                )
            except Exception as err:  # noqa: BLE001
                logger.warning(
                    "ignoring invalid rejected Lance metadata path={}: {}: {}",
                    rel_path,
                    type(err).__name__,
                    err,
                )
                continue
            rejected_created_files.extend(next_created_files)

        if not metadata_paths:
            self._cleanup_rejected_data(rejected_created_files)
            self._commit_empty_output(_import_lance())
            self._pending_metadata_cleanup = tuple(sorted(set(cleanup_paths)))
            return

        lance = _import_lance()
        fragment_json: list[str] = []
        schema: pa.Schema | None = None
        source_versions: set[int] = set()
        source_fragment_ids: set[int] = set()
        lance_schema_payload: dict[str, object] | None = None
        lance_schema: Any | None = None
        for rel_path in metadata_paths:
            (
                next_schema,
                next_fragments,
                next_created_files,
                next_source_version,
                next_source_fragment_ids,
                next_lance_schema,
            ) = self._read_metadata(rel_path)
            self._verified_created_files(
                lance,
                fragments=next_fragments,
                created_files=next_created_files,
                source_version=next_source_version,
                source_fragment_ids=next_source_fragment_ids,
                metadata_path=rel_path,
            )
            if next_schema is not None:
                if schema is None:
                    schema = next_schema
                elif not schema.equals(next_schema, check_metadata=True):
                    raise ValueError(
                        "Cannot commit Lance fragments with inconsistent schemas: "
                        + _schema_difference(schema, next_schema)
                    )
            fragment_json.extend(next_fragments)
            if next_source_version is not None:
                source_versions.add(next_source_version)
            for next_source_fragment_id in next_source_fragment_ids:
                if next_source_fragment_id in source_fragment_ids:
                    raise ValueError(
                        f"Duplicate Lance fragment result: {next_source_fragment_id}"
                    )
                source_fragment_ids.add(next_source_fragment_id)
            if next_lance_schema is not None:
                next_lance_schema_object = _lance_schema_from_payload(
                    lance, next_lance_schema
                )
                if lance_schema_payload is None:
                    lance_schema_payload = next_lance_schema
                    lance_schema = next_lance_schema_object
                elif lance_schema != next_lance_schema_object:
                    raise ValueError(
                        "Cannot commit Lance fragments with inconsistent field IDs."
                    )
        self._validate_add_columns_fragment_coverage(lance, source_fragment_ids)

        if not fragment_json:
            self._cleanup_rejected_data(rejected_created_files)
            self._commit_empty_output(lance)
            self._pending_metadata_cleanup = tuple(
                sorted(set(cleanup_paths).union(metadata_paths))
            )
            return
        if schema is None:
            raise ValueError("Lance fragment metadata is missing its schema")
        commit_message = self._commit_message(
            schema=schema,
            fragments=fragment_json,
            source_versions=source_versions,
            lance_schema_payload=lance_schema_payload,
        )
        expected_versions = (
            [next(iter(source_versions)) + 1]
            if self.mode in ("append", "overwrite") and len(source_versions) == 1
            else (
                [self.source_version + 1]
                if self.mode == "add_columns" and self.source_version is not None
                else ([1] if self.mode == "create" else [])
            )
        )
        if self._was_committed(
            lance, commit_message, expected_versions=expected_versions
        ):
            self._cleanup_rejected_data(rejected_created_files)
            self._pending_metadata_cleanup = tuple(
                sorted(set(cleanup_paths).union(metadata_paths))
            )
            return

        existing = self._load_existing_dataset(lance)
        if self.mode == "create" and existing is not None:
            raise ValueError(
                "Cannot create a Lance dataset at a location where one already exists."
            )
        if self.mode == "append":
            if existing is None:
                raise ValueError("Cannot append to a non-existent Lance dataset.")
            if len(source_versions) != 1:
                raise ValueError(
                    "Cannot append Lance fragments from different versions"
                )
            read_version = next(iter(source_versions))
            if existing.version != read_version:
                raise ValueError(
                    "Cannot append Lance fragments because the dataset changed "
                    f"from version {read_version} to {existing.version}"
                )
            operation = lance.LanceOperation.Append(
                [
                    lance.fragment.FragmentMetadata.from_json(fragment)
                    for fragment in fragment_json
                ]
            )
        elif self.mode == "add_columns":
            if existing is None:
                raise ValueError("Cannot add columns to a non-existent Lance dataset.")
            if self.source_version is None:
                raise ValueError("add_columns reducer is missing its source version")
            if source_versions != {self.source_version}:
                raise ValueError("Cannot merge Lance fragments from different versions")
            if existing.version != self.source_version:
                raise ValueError(
                    "Cannot add columns because the Lance dataset changed "
                    f"from version {self.source_version} to {existing.version}"
                )
            if lance_schema_payload is None:
                raise ValueError("add_columns metadata is missing the Lance schema")
            operation = lance.LanceOperation.Merge(
                [
                    lance.fragment.FragmentMetadata.from_json(fragment)
                    for fragment in fragment_json
                ],
                _lance_schema_from_payload(lance, lance_schema_payload),
            )
            read_version = self.source_version
        elif self.mode == "overwrite":
            if len(source_versions) != 1:
                raise ValueError(
                    "Cannot overwrite Lance fragments from different versions"
                )
            source_version = next(iter(source_versions))
            existing_version = int(existing.version) if existing is not None else 0
            if existing_version != source_version:
                raise ValueError(
                    "Cannot overwrite Lance fragments because the dataset changed "
                    f"from version {source_version} to {existing_version}"
                )
            operation = lance.LanceOperation.Overwrite(
                schema,
                [
                    lance.fragment.FragmentMetadata.from_json(fragment)
                    for fragment in fragment_json
                ],
            )
            read_version = source_version
        else:
            operation = lance.LanceOperation.Overwrite(
                schema,
                [
                    lance.fragment.FragmentMetadata.from_json(fragment)
                    for fragment in fragment_json
                ],
            )
            read_version = 0

        lance.LanceDataset.commit(
            self._dataset_uri(),
            operation,
            read_version=read_version,
            commit_message=commit_message,
            max_retries=0,
        )
        self._cleanup_rejected_data(rejected_created_files)
        self._pending_metadata_cleanup = tuple(
            sorted(set(cleanup_paths).union(metadata_paths))
        )

    def _cleanup_rejected_data(
        self,
        rejected_created_files: Sequence[str],
    ) -> None:
        _remove_paths_best_effort(
            self.output,
            rejected_created_files,
            operation="Lance rejected-file cleanup",
        )


__all__ = ["LanceDatasetCommitReducerSink", "LanceDatasetSink", "LanceWriteMode"]
