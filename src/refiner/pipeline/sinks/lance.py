from __future__ import annotations

import base64
import concurrent.futures
import json
import posixpath
import queue as queue_module
import re
from collections.abc import Sequence
from typing import Any, Literal, cast, get_args

import pyarrow as pa

from refiner.execution.asyncio.runtime import io_executor
from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.shard import (
    SHARD_ID_COLUMN,
    RowRangeDescriptor,
    Shard,
)
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.reducer.file import (
    FileCleanupReducerSink,
    _compile_output_path_patterns,
)
from refiner.pipeline.sources.lance import (
    LANCE_FRAGMENT_ID_COLUMN,
    LANCE_ROW_POSITION_COLUMN,
)
from refiner.worker.context import (
    get_active_stage_index,
    get_active_job_id,
    get_active_worker_token,
    get_finalized_workers,
)
from refiner.worker.metrics.api import log_throughput
from refiner.utils import check_required_dependencies

LanceWriteMode = Literal["create", "append", "overwrite", "add_columns"]
_METADATA_FILENAME_TEMPLATE = (
    "_refiner_lance_fragments/{job_id}/{shard_id}__w{worker_id}.jsonl"
)
_QUEUE_CLOSED = object()
_BATCH_QUEUE_SIZE = 8


def _import_lance() -> Any:
    check_required_dependencies(
        "write_lance_dataset", [("lance", "pylance")], dist="lance"
    )
    import lance

    return lance


def _import_lance_file_writer() -> Any:
    check_required_dependencies("write_lance", [("lance", "pylance")], dist="lance")
    from lance.file import LanceFileWriter

    return LanceFileWriter


def _validate_write_mode(mode: str) -> None:
    valid_modes = get_args(LanceWriteMode)
    if mode not in valid_modes:
        raise ValueError("mode must be one of: " + ", ".join(sorted(valid_modes)))


def _block_to_table(block: Block) -> pa.Table:
    table = (
        block.table
        if isinstance(block, Tabular)
        else (
            Tabular.from_rows(block).table
            if not block
            else block[0].tabular_type.from_rows(block).table
        )
    )
    if SHARD_ID_COLUMN in table.schema.names:
        table = table.drop_columns([SHARD_ID_COLUMN])
    return table


def _schema_to_base64(schema: pa.Schema) -> str:
    return base64.b64encode(schema.serialize().to_pybytes()).decode("ascii")


def _schema_from_base64(value: str) -> pa.Schema:
    return pa.ipc.read_schema(pa.BufferReader(base64.b64decode(value)))


def _lance_schema_to_payload(schema: Any) -> dict[str, object]:
    _, args = schema.__reduce__()
    metadata, *field_protos = args
    return {
        "metadata": str(metadata),
        "fields": [
            base64.b64encode(bytes(field_proto)).decode("ascii")
            for field_proto in field_protos
        ],
    }


def _lance_schema_from_payload(lance: Any, payload: object) -> Any:
    if not isinstance(payload, dict):
        raise ValueError("Invalid Lance schema metadata payload")
    payload_dict = cast(dict[str, object], payload)
    metadata = payload_dict.get("metadata")
    fields = payload_dict.get("fields")
    if not isinstance(metadata, str) or not isinstance(fields, list):
        raise ValueError("Invalid Lance schema metadata payload")
    encoded_fields: list[str] = []
    for field in fields:
        if not isinstance(field, str):
            raise ValueError("Invalid Lance schema field payload")
        encoded_fields.append(field)
    return lance.schema.LanceSchema._from_protos(
        metadata,
        *(base64.b64decode(field) for field in encoded_fields),
    )


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _metadata_prefix() -> str:
    return f"_refiner_lance_fragments/{get_active_job_id()}"


def _finalized_worker_pairs(*, reducer_name: str) -> set[tuple[str, str]]:
    stage_index = get_active_stage_index()
    if stage_index is None or stage_index <= 0:
        raise ValueError(
            f"{reducer_name} requires an active reducer stage with a prior writer stage"
        )
    return {
        (row.shard_id, row.worker_token)
        for row in get_finalized_workers(stage_index=stage_index - 1)
    }


def _managed_paths(
    *,
    output: DataFolder,
    managed_path_pattern: re.Pattern[str],
    search_path: str,
    reducer_name: str,
) -> tuple[list[str], list[str]]:
    keep_pairs = _finalized_worker_pairs(reducer_name=reducer_name)
    try:
        listed_paths = output.find(search_path)
    except FileNotFoundError:
        return [], []

    finalized_paths: list[str] = []
    cleanup_paths: list[str] = []
    for rel_path in listed_paths:
        if not isinstance(rel_path, str) or not rel_path or rel_path == ".":
            continue
        match = managed_path_pattern.fullmatch(rel_path)
        if match is None:
            continue
        cleanup_paths.append(rel_path)
        if (match.group("shard_id"), match.group("worker_id")) in keep_pairs:
            finalized_paths.append(rel_path)
    return finalized_paths, cleanup_paths


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


def _remove_fragment_data(output: DataFolder, fragment_json: str) -> None:
    for rel_path in _fragment_data_paths(fragment_json):
        try:
            output.rm(rel_path)
        except FileNotFoundError:
            continue


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
        self.queue_future: concurrent.futures.Future[
            queue_module.Queue[pa.RecordBatch | object]
        ] = concurrent.futures.Future()
        self.fragments: list[str] | None = None
        self.closed = False
        self.task_future = io_executor().submit(self._run)

    def _iter_batches(self):
        queue = self._queue()
        while True:
            item = queue.get()
            if item is _QUEUE_CLOSED:
                return
            yield item

    def _run(self) -> list[str]:
        queue: queue_module.Queue[pa.RecordBatch | object] = queue_module.Queue(
            maxsize=_BATCH_QUEUE_SIZE
        )
        self.queue_future.set_result(queue)
        lance = _import_lance()
        reader = pa.RecordBatchReader.from_batches(self.schema, self._iter_batches())
        fragments = lance.fragment.write_fragments(
            reader,
            self.dataset_uri,
            schema=self.schema,
            mode=self.mode,
        )
        fragments = [_json_dumps(fragment.to_json()) for fragment in fragments]
        self.fragments = fragments
        return fragments

    def _queue(self) -> queue_module.Queue[pa.RecordBatch | object]:
        return self.queue_future.result()

    def _raise_if_failed(self) -> None:
        if self.task_future is None or not self.task_future.done():
            return
        error = self.task_future.exception()
        if error is not None:
            raise RuntimeError("Lance fragment writer failed") from error

    def put_batches(self, batches: list[pa.RecordBatch]) -> None:
        if self.closed:
            raise RuntimeError("Cannot write to a closed Lance shard writer.")
        queue = self._queue()
        for batch in batches:
            self._raise_if_failed()
            queue.put(batch)
            self._raise_if_failed()

    def finish(self) -> list[str]:
        if not self.closed:
            self.closed = True
            self._raise_if_failed()
            self._queue().put(_QUEUE_CLOSED)
        if self.task_future is None:
            return []
        return list(self.task_future.result())


class _StreamingMergeColumnsWriter:
    def __init__(
        self,
        *,
        dataset_uri: str,
        version: int,
        fragment_id: int,
        schema: pa.Schema,
    ) -> None:
        self.dataset_uri = dataset_uri
        self.version = version
        self.fragment_id = fragment_id
        self.schema = schema
        self.queue_future: concurrent.futures.Future[
            queue_module.Queue[pa.RecordBatch | object]
        ] = concurrent.futures.Future()
        self.result: tuple[str, pa.Schema, dict[str, object], list[str]] | None = None
        self.closed = False
        self.task_future = io_executor().submit(self._run)

    def _iter_batches(self):
        queue = self._queue()
        while True:
            item = queue.get()
            if item is _QUEUE_CLOSED:
                return
            yield item

    def _run(self) -> tuple[str, pa.Schema, dict[str, object], list[str]]:
        queue: queue_module.Queue[pa.RecordBatch | object] = queue_module.Queue(
            maxsize=_BATCH_QUEUE_SIZE
        )
        self.queue_future.set_result(queue)
        lance = _import_lance()
        dataset = lance.dataset(self.dataset_uri, version=self.version)
        fragment = dataset.get_fragment(self.fragment_id)
        base_json = _json_dumps(fragment.metadata.to_json())
        reader = pa.RecordBatchReader.from_batches(self.schema, self._iter_batches())
        updated_fragment, merged_schema = fragment.merge_columns(
            reader,
            reader_schema=self.schema,
        )
        updated_json = _json_dumps(updated_fragment.to_json())
        created_files = sorted(
            set(_fragment_data_paths(updated_json)).difference(
                _fragment_data_paths(base_json)
            )
        )
        self.result = (
            updated_json,
            merged_schema.to_pyarrow(),
            _lance_schema_to_payload(merged_schema),
            created_files,
        )
        return self.result

    def _queue(self) -> queue_module.Queue[pa.RecordBatch | object]:
        return self.queue_future.result()

    def _raise_if_failed(self) -> None:
        if not self.task_future.done():
            return
        error = self.task_future.exception()
        if error is not None:
            raise RuntimeError("Lance add-columns writer failed") from error

    def put_batches(self, batches: list[pa.RecordBatch]) -> None:
        if self.closed:
            raise RuntimeError("Cannot write to a closed Lance shard writer.")
        queue = self._queue()
        for batch in batches:
            self._raise_if_failed()
            queue.put(batch)
            self._raise_if_failed()

    def finish(self) -> tuple[str, pa.Schema, dict[str, object], list[str]]:
        if not self.closed:
            self.closed = True
            self._raise_if_failed()
            self._queue().put(_QUEUE_CLOSED)
        return self.task_future.result()


class LanceSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}__w{worker_id}.lance",
    ) -> None:
        self.output = DataFolder.resolve(output)
        self.filename_template = filename_template
        self._writers: dict[str, Any] = {}

    def _declared_refiner_extras(self) -> tuple[str, ...]:
        return ("lance",)

    def _relpath(self, shard_id: str) -> str:
        return self.filename_template.format(
            shard_id=shard_id,
            worker_id=get_active_worker_token(),
        )

    def _writer(self, shard_id: str, schema: pa.Schema) -> Any:
        writer = self._writers.get(shard_id)
        if writer is not None:
            return writer
        LanceFileWriter = _import_lance_file_writer()
        writer = LanceFileWriter(self.output.abs_path(self._relpath(shard_id)), schema)
        self._writers[shard_id] = writer
        return writer

    def write_shard_block(self, shard_id: str, block: Block) -> None:
        table = _block_to_table(block)
        if table.num_rows == 0:
            return
        self._writer(shard_id, table.schema).write_batch(table)

    def on_shard_complete(self, shard_id: str) -> None:
        writer = self._writers.pop(shard_id, None)
        if writer is not None:
            writer.close()
            log_throughput("files_written", 1, shard_id=shard_id, unit="files")

    def close(self) -> None:
        first_error: Exception | None = None
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception as err:  # noqa: BLE001
                if first_error is None:
                    first_error = err
        self._writers.clear()
        if first_error is not None:
            raise first_error

    def describe(self) -> tuple[str, str, dict[str, object]]:
        return (
            "write_lance",
            "writer",
            {
                "path": self.output.abs_path(),
                "filename_template": self.filename_template,
            },
        )

    def build_reducer(self) -> BaseSink | None:
        return FileCleanupReducerSink(
            output=self.output,
            filename_template=self.filename_template,
            reducer_name="write_lance_reduce",
        )


class LanceDatasetSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        mode: LanceWriteMode = "create",
        columns: Sequence[str] | None = None,
        source_uri: str | None = None,
        source_version: int | None = None,
    ) -> None:
        _validate_write_mode(mode)
        if mode == "add_columns" and not columns:
            raise ValueError("add_columns requires at least one output column")
        if columns is not None and len(set(columns)) != len(columns):
            raise ValueError("Lance output columns must be unique")
        if mode != "add_columns" and columns is not None:
            raise ValueError("columns is only supported with mode='add_columns'")
        if mode == "add_columns" and (source_uri is None or source_version is None):
            raise ValueError("add_columns requires a version-pinned Lance source")
        self.output = DataFolder.resolve(output)
        self.mode = mode
        self.columns = tuple(columns) if columns is not None else None
        self.source_uri = source_uri
        self.source_version = source_version
        if mode == "add_columns" and self.output.abs_path() != source_uri:
            raise ValueError("add_columns must write back to the loaded Lance dataset")
        self._writers_by_shard: dict[
            str, _StreamingShardWriter | _StreamingMergeColumnsWriter
        ] = {}
        self._schema_by_shard: dict[str, pa.Schema] = {}
        self._source_fragment_by_shard: dict[str, tuple[int, int]] = {}
        self._next_position_by_shard: dict[str, int] = {}
        self._pending_rows_by_shard: dict[str, dict[int, dict[str, Any]]] = {}
        self._add_columns_schema: pa.Schema | None = None
        self._existing_schema: pa.Schema | None = None

    def _declared_refiner_extras(self) -> tuple[str, ...]:
        return ("lance",)

    def _dataset_uri(self) -> str:
        return self.output.abs_path()

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
        return self._existing_schema

    def set_input_schema(self, schema: pa.Schema | None) -> None:
        if self.mode != "add_columns" or schema is None:
            return
        assert self.columns is not None
        missing = sorted(set(self.columns).difference(schema.names))
        if missing:
            raise ValueError(
                "add_columns outputs are missing from the pipeline schema: "
                + ", ".join(missing)
            )
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
        source_schema = (
            _import_lance()
            .dataset(
                self.source_uri,
                version=self.source_version,
            )
            .schema
        )
        conflicts = sorted(set(self.columns).intersection(source_schema.names))
        if conflicts:
            raise ValueError(
                "add_columns cannot replace existing columns: " + ", ".join(conflicts)
            )
        self._add_columns_schema = pa.schema(
            [schema.field(column) for column in self.columns]
        )

    def on_shard_start(self, shard: Shard) -> None:
        if self.mode != "add_columns":
            return
        descriptor = shard.descriptor
        if not isinstance(descriptor, RowRangeDescriptor):
            raise TypeError("add_columns requires row-range shards")
        if descriptor.end != descriptor.start + 1:
            raise ValueError("Lance shards must identify exactly one fragment")
        assert self.source_uri is not None
        assert self.source_version is not None
        fragments = (
            _import_lance()
            .dataset(
                self.source_uri,
                version=self.source_version,
            )
            .get_fragments()
        )
        if descriptor.start < 0 or descriptor.start >= len(fragments):
            raise ValueError(
                f"Lance fragment index {descriptor.start} is out of bounds"
            )
        fragment = fragments[descriptor.start]
        self._source_fragment_by_shard[shard.id] = (
            int(fragment.fragment_id),
            int(fragment.count_rows()),
        )
        self._next_position_by_shard[shard.id] = 0
        self._pending_rows_by_shard[shard.id] = {}

    def _add_columns_writer(
        self,
        shard_id: str,
        schema: pa.Schema,
    ) -> _StreamingMergeColumnsWriter:
        writer = self._writers_by_shard.get(shard_id)
        if writer is not None:
            if not isinstance(writer, _StreamingMergeColumnsWriter):
                raise TypeError("Invalid writer for add_columns")
            return writer
        fragment_metadata = self._source_fragment_by_shard.get(shard_id)
        if fragment_metadata is None:
            raise ValueError(f"Missing Lance source metadata for shard {shard_id}")
        fragment_id, _ = fragment_metadata
        assert self.source_uri is not None
        assert self.source_version is not None
        writer = _StreamingMergeColumnsWriter(
            dataset_uri=self.source_uri,
            version=self.source_version,
            fragment_id=fragment_id,
            schema=schema,
        )
        self._writers_by_shard[shard_id] = writer
        return writer

    def _write_add_columns_block(self, shard_id: str, block: Block) -> None:
        table = _block_to_table(block)
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

        fragment_metadata = self._source_fragment_by_shard.get(shard_id)
        if fragment_metadata is None:
            raise ValueError(f"Missing Lance source metadata for shard {shard_id}")
        fragment_id, num_rows = fragment_metadata
        next_position = self._next_position_by_shard[shard_id]
        pending = self._pending_rows_by_shard[shard_id]
        for row in table.to_pylist():
            raw_fragment_id = row[LANCE_FRAGMENT_ID_COLUMN]
            if not isinstance(raw_fragment_id, int):
                raise ValueError("Lance fragment id must be an integer")
            if int(raw_fragment_id) != fragment_id:
                raise ValueError(
                    f"Lance row belongs to fragment {raw_fragment_id}; "
                    f"expected {fragment_id}"
                )
            raw_position = row[LANCE_ROW_POSITION_COLUMN]
            if not isinstance(raw_position, int):
                raise ValueError("Lance row position must be an integer")
            position = int(raw_position)
            if position < 0 or position >= num_rows:
                raise ValueError(
                    f"Lance row position {position} is outside fragment bounds"
                )
            if position < next_position or position in pending:
                raise ValueError(
                    f"Duplicate Lance row position {position} in shard {shard_id}"
                )
            pending[position] = {column: row[column] for column in self.columns}

        contiguous: list[dict[str, Any]] = []
        while next_position in pending:
            contiguous.append(pending.pop(next_position))
            next_position += 1
        self._next_position_by_shard[shard_id] = next_position

        if contiguous:
            output = pa.Table.from_pylist(contiguous, schema=self._add_columns_schema)
            self._add_columns_writer(
                shard_id,
                self._add_columns_schema,
            ).put_batches(output.to_batches())

    def write_shard_block(self, shard_id: str, block: Block) -> None:
        if self.mode == "add_columns":
            self._write_add_columns_block(shard_id, block)
            return
        table = _block_to_table(block)
        if table.num_rows == 0:
            return
        if self.mode == "append":
            table = table.cast(self._load_existing_schema())

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
        payload = {
            "schema": _schema_to_base64(schema),
            "fragments": fragments,
        }
        with self.output.open(
            self._relpath(shard_id), mode="wt", encoding="utf-8"
        ) as f:
            f.write(_json_dumps(payload))
            f.write("\n")
        log_throughput("files_written", 1, shard_id=shard_id, unit="files")

    def _complete_add_columns_shard(self, shard_id: str) -> None:
        fragment_metadata = self._source_fragment_by_shard.pop(shard_id, None)
        pending = self._pending_rows_by_shard.pop(shard_id, {})
        written = self._next_position_by_shard.pop(shard_id, 0)
        if fragment_metadata is None:
            raise ValueError(f"Missing Lance source metadata for shard {shard_id}")
        fragment_id, num_rows = fragment_metadata
        if pending or written != num_rows:
            raise ValueError(
                f"Lance fragment {fragment_id} produced {written} contiguous "
                f"rows out of {num_rows}"
            )
        writer = self._writers_by_shard.pop(shard_id, None)
        if not isinstance(writer, _StreamingMergeColumnsWriter):
            if num_rows == 0:
                return
            raise ValueError(f"Lance fragment {fragment_id} produced no output")
        updated_fragment, schema, lance_schema, created_files = writer.finish()
        payload = {
            "schema": _schema_to_base64(schema),
            "lance_schema": lance_schema,
            "fragments": [updated_fragment],
            "created_files": created_files,
            "source_version": self.source_version,
            "source_fragment_id": fragment_id,
        }
        with self.output.open(
            self._relpath(shard_id), mode="wt", encoding="utf-8"
        ) as f:
            f.write(_json_dumps(payload))
            f.write("\n")
        log_throughput("files_written", 1, shard_id=shard_id, unit="files")

    def close(self) -> None:
        first_error: Exception | None = None
        for writer in self._writers_by_shard.values():
            try:
                if isinstance(writer, _StreamingMergeColumnsWriter):
                    _, _, _, created_files = writer.finish()
                    for path in created_files:
                        try:
                            self.output.rm(path)
                        except FileNotFoundError:
                            continue
                else:
                    for fragment in writer.finish():
                        _remove_fragment_data(self.output, fragment)
            except Exception as err:  # noqa: BLE001
                if first_error is None:
                    first_error = err
        self._writers_by_shard.clear()
        self._schema_by_shard.clear()
        self._source_fragment_by_shard.clear()
        self._next_position_by_shard.clear()
        self._pending_rows_by_shard.clear()
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
        return LanceDatasetCommitReducerSink(
            self.output,
            mode=self.mode,
            source_version=self.source_version,
        )


class LanceDatasetCommitReducerSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        mode: LanceWriteMode,
        source_version: int | None = None,
    ) -> None:
        _validate_write_mode(mode)
        self.output = DataFolder.resolve(output)
        self.mode = mode
        self.source_version = source_version
        self._managed_path_pattern = _compile_output_path_patterns(
            _METADATA_FILENAME_TEMPLATE
        )[-1]
        self._commit_ran = False

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
        if not isinstance(schema_raw, str) or not isinstance(fragment_raw, list):
            raise ValueError(f"Invalid Lance metadata payload: {rel_path}")
        if not isinstance(created_raw, list):
            raise ValueError(f"Invalid Lance created-files payload: {rel_path}")
        return (
            _schema_from_base64(schema_raw),
            [str(fragment) for fragment in fragment_raw],
            [str(path) for path in created_raw],
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

    def _run_commit(self) -> None:
        if self._commit_ran:
            return
        self._commit_ran = True

        metadata_paths, cleanup_paths = _managed_paths(
            output=self.output,
            managed_path_pattern=self._managed_path_pattern,
            search_path=_metadata_prefix(),
            reducer_name="write_lance_dataset_commit",
        )
        rejected_paths = sorted(set(cleanup_paths).difference(metadata_paths))

        rejected_fragments: list[str] = []
        rejected_created_files: list[str] = []
        for rel_path in rejected_paths:
            (
                _,
                next_rejected_fragments,
                next_created_files,
                _,
                _,
                _,
            ) = self._read_metadata(rel_path)
            rejected_fragments.extend(next_rejected_fragments)
            rejected_created_files.extend(next_created_files)

        if not metadata_paths:
            self._cleanup_rejected_data(
                rejected_fragments,
                rejected_created_files,
            )
            for rel_path in cleanup_paths:
                try:
                    self.output.rm(rel_path)
                except FileNotFoundError:
                    continue
            return

        lance = _import_lance()
        fragment_json: list[str] = []
        schema: pa.Schema | None = None
        source_versions: set[int] = set()
        source_fragment_ids: set[int] = set()
        lance_schema_payload: dict[str, object] | None = None
        for rel_path in sorted(metadata_paths):
            (
                next_schema,
                next_fragments,
                _,
                next_source_version,
                next_source_fragment_id,
                next_lance_schema,
            ) = self._read_metadata(rel_path)
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
                        f"Duplicate Lance fragment result: {next_source_fragment_id}"
                    )
                source_fragment_ids.add(next_source_fragment_id)
            if next_lance_schema is not None:
                if lance_schema_payload is None:
                    lance_schema_payload = next_lance_schema
                elif lance_schema_payload != next_lance_schema:
                    raise ValueError(
                        "Cannot commit Lance fragments with inconsistent field IDs."
                    )

        if schema is None or not fragment_json:
            return

        existing = self._load_existing_dataset(lance)
        if self.mode == "create" and existing is not None:
            raise ValueError(
                "Cannot create a Lance dataset at a location where one already exists."
            )
        if self.mode == "append":
            if existing is None:
                raise ValueError("Cannot append to a non-existent Lance dataset.")
            operation = lance.LanceOperation.Append(
                [
                    lance.fragment.FragmentMetadata.from_json(fragment)
                    for fragment in fragment_json
                ]
            )
            read_version = existing.version
        elif self.mode == "add_columns":
            if existing is None:
                raise ValueError("Cannot add columns to a non-existent Lance dataset.")
            if self.source_version is None:
                raise ValueError("add_columns reducer is missing its source version")
            if source_versions != {self.source_version}:
                raise ValueError("Cannot merge Lance fragments from different versions")
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
        else:
            operation = lance.LanceOperation.Overwrite(
                schema,
                [
                    lance.fragment.FragmentMetadata.from_json(fragment)
                    for fragment in fragment_json
                ],
            )
            read_version = existing.version if existing is not None else 0

        commit_options = {"max_retries": 0} if self.mode == "add_columns" else {}
        lance.LanceDataset.commit(
            self._dataset_uri(),
            operation,
            read_version=read_version,
            **commit_options,
        )
        self._cleanup_rejected_data(
            rejected_fragments,
            rejected_created_files,
        )
        for rel_path in cleanup_paths:
            try:
                self.output.rm(rel_path)
            except FileNotFoundError:
                continue

    def _cleanup_rejected_data(
        self,
        rejected_fragments: Sequence[str],
        rejected_created_files: Sequence[str],
    ) -> None:
        if self.mode == "add_columns":
            for path in rejected_created_files:
                try:
                    self.output.rm(path)
                except FileNotFoundError:
                    continue
            return
        for fragment in rejected_fragments:
            _remove_fragment_data(self.output, fragment)


__all__ = ["LanceDatasetCommitReducerSink", "LanceDatasetSink", "LanceSink"]
