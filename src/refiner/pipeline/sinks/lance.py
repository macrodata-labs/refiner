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
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.shard import SHARD_ID_COLUMN
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.lance_schema import (
    lance_schema_from_payload as _lance_schema_from_payload,
    lance_schema_to_payload as _lance_schema_to_payload,
)
from refiner.pipeline.sinks.lance_utils import block_to_table, validate_lance_uri
from refiner.pipeline.sinks.reducer.file import (
    _compile_output_path_patterns,
)
from refiner.pipeline.sources.lance import (
    LANCE_FRAGMENT_ID_COLUMN,
    LANCE_INTERNAL_COLUMNS,
    LANCE_ROW_POSITION_COLUMN,
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


def _import_lance() -> Any:
    check_required_dependencies(
        "write_lance_dataset", [("lance", "pylance")], dist="lance"
    )
    import lance

    return lance


def _validate_write_mode(mode: str) -> None:
    valid_modes = get_args(LanceWriteMode)
    if mode not in valid_modes:
        raise ValueError("mode must be one of: " + ", ".join(sorted(valid_modes)))


def _schema_to_base64(schema: pa.Schema) -> str:
    return base64.b64encode(schema.serialize().to_pybytes()).decode("ascii")


def _schema_from_base64(value: str) -> pa.Schema:
    return pa.ipc.read_schema(pa.BufferReader(base64.b64decode(value)))


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _metadata_prefix() -> str:
    return f"_refiner_lance_fragments/{get_active_job_id()}"


def _finalized_workers(*, reducer_name: str) -> list[Any]:
    stage_index = get_active_stage_index()
    if stage_index is None or stage_index <= 0:
        raise ValueError(
            f"{reducer_name} requires an active reducer stage with a prior writer stage"
        )
    return sort_finalized_workers(
        get_finalized_workers(stage_index=stage_index - 1)
    )


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
        (row.shard_id, row.worker_token): row.job_id
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
    ) -> None:
        self.fragment = fragment
        self.fragment_id = fragment_id
        self.num_rows = num_rows
        self.schema = schema
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
            tmp = tempfile.NamedTemporaryFile(prefix="refiner-lance-columns-", delete=False)
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
            self.pending[start] = (end, sorted_output.slice(run_start, index - run_start))
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

    def finish(self) -> tuple[Any, Any]:
        complete = self.next_position == self.num_rows and not self.pending
        self._close_input()
        if not complete:
            if self._spool_path is None:
                try:
                    self.task_future.result()
                except Exception:  # noqa: BLE001
                    pass
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
            self.task_future.result()
        except Exception:  # noqa: BLE001
            pass


class LanceDatasetSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        mode: LanceWriteMode = "create",
        columns: Sequence[str] | None = None,
        source_uri: str | None = None,
        source_version: int | None = None,
        planned_schema: pa.Schema | None = None,
    ) -> None:
        _validate_write_mode(mode)
        if mode == "add_columns" and not columns:
            raise ValueError("add_columns requires at least one output column")
        if columns is not None and len(set(columns)) != len(columns):
            raise ValueError("Lance output columns must be unique")
        if columns is not None:
            reserved_columns = {SHARD_ID_COLUMN, *LANCE_INTERNAL_COLUMNS}.intersection(
                columns
            )
            if reserved_columns:
                raise ValueError(f"{sorted(reserved_columns)[0]} is an internal column")
        if mode != "add_columns" and columns is not None:
            raise ValueError("columns is only supported with mode='add_columns'")
        if mode == "add_columns" and (source_uri is None or source_version is None):
            raise ValueError("add_columns requires a version-pinned Lance source")
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
        self.planned_schema = planned_schema
        if mode == "add_columns" and self.output.abs_path() != source_uri:
            raise ValueError("add_columns must write back to the loaded Lance dataset")
        self._writers_by_shard: dict[str, _StreamingShardWriter] = {}
        self._schema_by_shard: dict[str, pa.Schema] = {}
        self._add_columns_writers_by_shard: dict[
            str, _StreamingAddColumnsWriter
        ] = {}
        self._add_columns_schema: pa.Schema | None = None
        self._existing_schema: pa.Schema | None = None
        self._existing_version: int | None = None
        self._source_dataset_cache: Any | None = None

    def _declared_refiner_extras(self) -> tuple[str, ...]:
        return ("lance",)

    @property
    def retained_source_columns(self) -> frozenset[str]:
        return LANCE_INTERNAL_COLUMNS if self.mode == "add_columns" else frozenset()

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
            job_id=get_active_job_id(),
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
        if self.mode != "add_columns" or schema is None:
            return
        assert self.columns is not None
        missing = sorted(set(self.columns).difference(schema.names))
        for internal_column in (
            LANCE_FRAGMENT_ID_COLUMN,
            LANCE_ROW_POSITION_COLUMN,
        ):
            if internal_column not in schema.names:
                raise ValueError(
                    f"add_columns requires internal column {internal_column}"
                )
        assert self.source_uri is not None
        assert self.source_version is not None
        source_schema = self._source_dataset().schema
        conflicts = sorted(set(self.columns).intersection(source_schema.names))
        if conflicts:
            raise ValueError(
                "add_columns cannot replace existing columns: " + ", ".join(conflicts)
            )
        if not missing:
            self._add_columns_schema = pa.schema(
                [schema.field(column) for column in self.columns]
            )

    def _write_add_columns_block(self, shard_id: str, block: Block) -> None:
        table = block_to_table(block)
        if table.num_rows == 0:
            return
        for internal_column in (
            LANCE_FRAGMENT_ID_COLUMN,
            LANCE_ROW_POSITION_COLUMN,
        ):
            if internal_column not in table.schema.names:
                raise ValueError(f"add_columns output is missing {internal_column}")
        assert self.columns is not None
        missing = sorted(set(self.columns).difference(table.schema.names))
        if missing:
            raise ValueError(
                "add_columns output is missing columns: " + ", ".join(missing)
            )

        output_schema = pa.schema(
            [table.schema.field(column) for column in self.columns]
        )
        if self._add_columns_schema is None:
            self._add_columns_schema = output_schema
        elif not self._add_columns_schema.equals(output_schema):
            raise ValueError("add_columns output schema changed between blocks")
        fragment_ids = pc.cast(
            table.column(LANCE_FRAGMENT_ID_COLUMN), pa.uint64()
        )
        if fragment_ids.null_count:
            raise ValueError("Lance fragment id cannot be null")
        fragment_id_range = pc.call_function("min_max", [fragment_ids]).as_py()
        if fragment_id_range["min"] != fragment_id_range["max"]:
            raise ValueError(f"Shard {shard_id} contains multiple Lance fragments")
        fragment_id = int(fragment_id_range["min"])

        writer = self._add_columns_writers_by_shard.get(shard_id)
        if writer is None:
            fragment = self._source_dataset().get_fragment(fragment_id)
            writer = _StreamingAddColumnsWriter(
                fragment=fragment,
                fragment_id=fragment_id,
                num_rows=int(fragment.count_rows()),
                schema=output_schema,
            )
            self._add_columns_writers_by_shard[shard_id] = writer
        elif writer.fragment_id != fragment_id:
            raise ValueError(f"Shard {shard_id} contains multiple Lance fragments")
        writer.put(
            table.column(LANCE_ROW_POSITION_COLUMN),
            table.select(self.columns),
        )

    def write_shard_block(self, shard_id: str, block: Block) -> None:
        if self.mode == "add_columns":
            self._write_add_columns_block(shard_id, block)
            return
        table = block_to_table(block)
        if table.num_rows == 0:
            return
        if self.mode == "append":
            table = table.cast(self._load_existing_schema())
        elif self.mode == "overwrite":
            self._load_overwrite_version()

        existing_schema = self._schema_by_shard.setdefault(shard_id, table.schema)
        if not existing_schema.equals(table.schema):
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

    def on_shard_complete(self, shard_id: str) -> None:
        if self.mode == "add_columns":
            self._complete_add_columns_shard(shard_id)
            return
        writer = self._writers_by_shard.pop(shard_id, None)
        schema = self._schema_by_shard.pop(shard_id, None)
        if writer is None or schema is None:
            return
        fragments = writer.finish()
        if not fragments:
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
        try:
            with self.output.open(
                self._relpath(shard_id), mode="wt", encoding="utf-8"
            ) as f:
                f.write(_json_dumps(payload))
                f.write("\n")
        except Exception:
            for fragment in fragments:
                _remove_fragment_data(self.output, fragment)
            raise
        log_throughput("files_written", 1, shard_id=shard_id, unit="files")

    def _complete_add_columns_shard(self, shard_id: str) -> None:
        writer = self._add_columns_writers_by_shard.pop(shard_id, None)
        if writer is None:
            return
        assert self.source_version is not None
        updated_fragment, merged_schema = writer.finish()
        updated_json = _json_dumps(updated_fragment.to_json())
        created_files = sorted(
            set(_fragment_data_paths(updated_json)).difference(
                _fragment_data_paths(writer.base_json)
            )
        )
        payload = {
            "schema": _schema_to_base64(merged_schema.to_pyarrow()),
            "lance_schema": _lance_schema_to_payload(merged_schema),
            "fragments": [updated_json],
            "created_files": created_files,
            "source_version": self.source_version,
            "source_fragment_id": writer.fragment_id,
        }
        try:
            with self.output.open(
                self._relpath(shard_id), mode="wt", encoding="utf-8"
            ) as f:
                f.write(_json_dumps(payload))
                f.write("\n")
        except Exception:
            for path in created_files:
                try:
                    self.output.rm(path)
                except FileNotFoundError:
                    continue
            raise
        log_throughput("files_written", 1, shard_id=shard_id, unit="files")

    def close(self) -> None:
        first_error: Exception | None = None
        for writer in self._writers_by_shard.values():
            try:
                for fragment in writer.finish():
                    _remove_fragment_data(self.output, fragment)
            except Exception as err:  # noqa: BLE001
                if first_error is None:
                    first_error = err
        self._writers_by_shard.clear()
        self._schema_by_shard.clear()
        for writer in self._add_columns_writers_by_shard.values():
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
        return ("write_lance_dataset", "writer", args)

    def build_reducer(self) -> BaseSink | None:
        source_version = self.source_version
        if self.mode == "overwrite":
            source_version = self._load_overwrite_version()
        return LanceDatasetCommitReducerSink(
            self.output,
            mode=self.mode,
            source_version=source_version,
            planned_schema=self.planned_schema,
        )


class LanceDatasetCommitReducerSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        mode: LanceWriteMode,
        source_version: int | None = None,
        planned_schema: pa.Schema | None = None,
    ) -> None:
        _validate_write_mode(mode)
        self.output = DataFolder.resolve(output)
        validate_lance_uri(self.output.abs_path())
        self.mode = mode
        self.source_version = source_version
        self.planned_schema = planned_schema
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
        pa.Schema,
        list[str],
        list[str],
        int | None,
        int | None,
        dict[str, object] | None,
    ]:
        with self.output.open(rel_path, mode="rt", encoding="utf-8") as f:
            payload = json.load(f)
        schema_raw = payload.get("schema") if isinstance(payload, dict) else None
        fragment_raw = payload.get("fragments") if isinstance(payload, dict) else None
        created_raw = (
            payload.get("created_files", []) if isinstance(payload, dict) else []
        )
        source_version_raw = (
            payload.get("source_version") if isinstance(payload, dict) else None
        )
        source_fragment_raw = (
            payload.get("source_fragment_id") if isinstance(payload, dict) else None
        )
        lance_schema_raw = (
            payload.get("lance_schema") if isinstance(payload, dict) else None
        )
        if (
            not isinstance(schema_raw, str)
            or not isinstance(fragment_raw, list)
            or not all(isinstance(fragment, str) for fragment in fragment_raw)
        ):
            raise ValueError(f"Invalid Lance metadata payload: {rel_path}")
        if not isinstance(created_raw, list):
            raise ValueError(f"Invalid Lance created-files payload: {rel_path}")
        fragments = [fragment for fragment in fragment_raw if isinstance(fragment, str)]
        return (
            _schema_from_base64(schema_raw),
            fragments,
            [_validate_created_file_path(path) for path in created_raw],
            int(source_version_raw) if source_version_raw is not None else None,
            int(source_fragment_raw) if source_fragment_raw is not None else None,
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
        source_fragment_id: int | None,
        metadata_path: str,
    ) -> list[str]:
        if self._managed_path_pattern.fullmatch(metadata_path) is None:
            raise ValueError(f"Invalid Lance metadata path: {metadata_path}")
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
            or source_fragment_id is None
            or len(fragments) != 1
        ):
            raise ValueError("Invalid rejected Lance add-columns metadata")
        base_fragment = lance.dataset(
            self._dataset_uri(), version=source_version
        ).get_fragment(source_fragment_id)
        base_json = _json_dumps(base_fragment.metadata.to_json())
        expected = set(_fragment_data_paths(fragments[0])).difference(
            _fragment_data_paths(base_json)
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
        if self.planned_schema is None:
            raise ValueError(
                f"Cannot {self.mode} an empty Lance dataset without a known schema"
            )
        if self.mode == "overwrite" and self.source_version is None:
            raise ValueError("Empty Lance overwrite is missing its source version")
        finalized = _finalized_workers(reducer_name="write_lance_dataset_commit")
        commit_message = self._commit_message(
            schema=self.planned_schema,
            fragments=[f"{row.shard_id}/{row.worker_token}" for row in finalized],
            source_versions=set(),
            lance_schema_payload=None,
        )
        expected_version = 1 if self.mode == "create" else self.source_version + 1
        if self._was_committed(
            lance,
            commit_message,
            expected_versions=[expected_version],
            search_history=self.mode == "overwrite",
        ):
            return
        if self.mode == "create" and existing is not None:
            raise ValueError(
                "Cannot create a Lance dataset at a location where one already exists."
            )
        existing_version = int(existing.version) if existing is not None else 0
        if self.mode == "overwrite" and existing_version != self.source_version:
            raise ValueError(
                "Cannot overwrite an empty Lance dataset because the dataset changed "
                f"from version {self.source_version} to {existing_version}"
            )
        operation = lance.LanceOperation.Overwrite(self.planned_schema, [])
        lance.LanceDataset.commit(
            self._dataset_uri(),
            operation,
            read_version=existing_version,
            max_retries=0,
            commit_message=commit_message,
        )

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
                    next_source_fragment_id,
                    _,
                ) = self._read_metadata(rel_path)
                next_created_files = self._verified_created_files(
                    cleanup_lance,
                    fragments=next_rejected_fragments,
                    created_files=next_created_files,
                    source_version=next_source_version,
                    source_fragment_id=next_source_fragment_id,
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
        for rel_path in metadata_paths:
            (
                next_schema,
                next_fragments,
                next_created_files,
                next_source_version,
                next_source_fragment_id,
                next_lance_schema,
            ) = self._read_metadata(rel_path)
            self._verified_created_files(
                lance,
                fragments=next_fragments,
                created_files=next_created_files,
                source_version=next_source_version,
                source_fragment_id=next_source_fragment_id,
                metadata_path=rel_path,
            )
            if schema is None:
                schema = next_schema
            elif not schema.equals(next_schema):
                raise ValueError(
                    "Cannot commit Lance fragments with inconsistent schemas."
                )
            fragment_json.extend(next_fragments)
            if next_source_version is not None:
                source_versions.add(next_source_version)
            if next_source_fragment_id is not None:
                if next_source_fragment_id in source_fragment_ids:
                    raise ValueError(
                        "Duplicate Lance fragment result: "
                        f"{next_source_fragment_id}"
                    )
                source_fragment_ids.add(next_source_fragment_id)
            if next_lance_schema is not None:
                if lance_schema_payload is None:
                    lance_schema_payload = next_lance_schema
                elif lance_schema_payload != next_lance_schema:
                    raise ValueError(
                        "Cannot commit Lance fragments with inconsistent field IDs."
                    )
        self._validate_add_columns_fragment_coverage(lance, source_fragment_ids)

        if schema is None or not fragment_json:
            return
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
