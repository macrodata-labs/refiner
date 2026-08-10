from __future__ import annotations

import base64
import json
import posixpath
import queue as queue_module
import re
from collections.abc import Sequence
from typing import Any, Literal, cast, get_args

import pyarrow as pa
import pyarrow.compute as pc

from refiner.execution.asyncio.runtime import io_executor
from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.shard import SHARD_ID_COLUMN
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.lance_file import LanceSink
from refiner.pipeline.sinks.reducer.file import (
    _compile_output_path_patterns,
)
from refiner.pipeline.sources.lance import (
    LANCE_FRAGMENT_ID_COLUMN,
    LANCE_ROW_POSITION_COLUMN,
    _open_lance_dataset,
)
from refiner.worker.context import (
    get_active_stage_index,
    get_active_job_id,
    get_active_worker_token,
    get_finalized_workers,
    logger,
)
from refiner.worker.metrics.api import log_throughput
from refiner.utils import check_required_dependencies

LanceWriteMode = Literal["create", "append", "overwrite", "add_columns"]
_METADATA_FILENAME_TEMPLATE = (
    "_refiner_lance_fragments/{job_id}/{shard_id}__w{worker_id}.jsonl"
)
_QUEUE_CLOSED = object()


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
    cleanup_path_prefix: str | None = None,
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
        if cleanup_path_prefix is None or rel_path.startswith(
            cleanup_path_prefix.rstrip("/") + "/"
        ):
            cleanup_paths.append(rel_path)
        if (match.group("shard_id"), match.group("worker_id")) in keep_pairs:
            finalized_paths.append(rel_path)
    return sorted(set(finalized_paths)), sorted(set(cleanup_paths))


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
        self.queue: queue_module.Queue[pa.RecordBatch | object] = queue_module.Queue()
        self.closed = False
        self.task_future = io_executor().submit(self._run)

    def _iter_batches(self):
        while True:
            item = self.queue.get()
            if item is _QUEUE_CLOSED:
                return
            yield item

    def _run(self) -> list[str]:
        lance = _import_lance()
        reader = pa.RecordBatchReader.from_batches(self.schema, self._iter_batches())
        fragments = lance.fragment.write_fragments(
            reader,
            self.dataset_uri,
            schema=self.schema,
            mode=self.mode,
        )
        return [_json_dumps(fragment.to_json()) for fragment in fragments]

    def _raise_if_failed(self) -> None:
        if not self.task_future.done():
            return
        error = self.task_future.exception()
        if error is not None:
            raise RuntimeError("Lance fragment writer failed") from error

    def put_batches(self, batches: list[pa.RecordBatch]) -> None:
        if self.closed:
            raise RuntimeError("Cannot write to a closed Lance shard writer.")
        for batch in batches:
            self._raise_if_failed()
            self.queue.put(batch)
            self._raise_if_failed()

    def finish(self) -> list[str]:
        if not self.closed:
            self.closed = True
            self._raise_if_failed()
            self.queue.put(_QUEUE_CLOSED)
        return list(self.task_future.result())


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
        self.mode = mode
        self.columns = tuple(columns) if columns is not None else None
        self.source_uri = source_uri
        self.source_version = source_version
        self.planned_schema = planned_schema
        if mode == "add_columns" and self.output.abs_path() != source_uri:
            raise ValueError("add_columns must write back to the loaded Lance dataset")
        self._writers_by_shard: dict[str, _StreamingShardWriter] = {}
        self._schema_by_shard: dict[str, pa.Schema] = {}
        self._add_columns_tables_by_shard: dict[str, list[pa.Table]] = {}
        self._add_columns_schema: pa.Schema | None = None
        self._existing_schema: pa.Schema | None = None
        self._existing_version: int | None = None

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
        self._existing_version = int(dataset.version)
        return self._existing_schema

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
        source_schema = _open_lance_dataset(
            self.source_uri,
            self.source_version,
        ).schema
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
        buffered = table.select(
            [
                LANCE_FRAGMENT_ID_COLUMN,
                LANCE_ROW_POSITION_COLUMN,
                *self.columns,
            ]
        )
        self._add_columns_tables_by_shard.setdefault(shard_id, []).append(buffered)

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
        payload: dict[str, object] = {
            "schema": _schema_to_base64(schema),
            "fragments": fragments,
        }
        if self.mode == "append":
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
        tables = self._add_columns_tables_by_shard.pop(shard_id, None)
        if not tables:
            return
        assert self.columns is not None
        assert self.source_uri is not None
        assert self.source_version is not None
        assert self._add_columns_schema is not None

        table = pa.concat_tables(tables)
        fragment_ids = table.column(LANCE_FRAGMENT_ID_COLUMN)
        if fragment_ids.null_count:
            raise ValueError("Lance fragment id cannot be null")
        fragment_id_range = pc.call_function("min_max", [fragment_ids]).as_py()
        if fragment_id_range["min"] != fragment_id_range["max"]:
            raise ValueError(f"Shard {shard_id} contains multiple Lance fragments")
        fragment_id = int(fragment_id_range["min"])

        fragment = _open_lance_dataset(
            self.source_uri, self.source_version
        ).get_fragment(fragment_id)
        num_rows = int(fragment.count_rows())
        if table.num_rows != num_rows:
            raise ValueError(
                f"Lance fragment {fragment_id} produced {table.num_rows} "
                f"rows out of {num_rows}"
            )

        positions = table.column(LANCE_ROW_POSITION_COLUMN)
        if positions.null_count:
            raise ValueError("Lance row position cannot be null")
        expected_positions = pa.array(range(num_rows), type=pa.uint64())
        positions_match = positions.combine_chunks().equals(expected_positions)
        output = table.select(self.columns)
        if not positions_match:
            indices = pc.call_function("sort_indices", [positions])
            sorted_positions = pc.take(positions, indices)
            if not sorted_positions.combine_chunks().equals(expected_positions):
                raise ValueError(
                    f"Lance fragment {fragment_id} has missing or duplicate row positions"
                )
            output = output.take(indices)

        base_json = _json_dumps(fragment.metadata.to_json())
        reader = pa.RecordBatchReader.from_batches(
            self._add_columns_schema,
            output.to_batches(),
        )
        updated_fragment, merged_schema = fragment.merge_columns(
            reader,
            reader_schema=self._add_columns_schema,
        )
        updated_json = _json_dumps(updated_fragment.to_json())
        created_files = sorted(
            set(_fragment_data_paths(updated_json)).difference(
                _fragment_data_paths(base_json)
            )
        )
        payload = {
            "schema": _schema_to_base64(merged_schema.to_pyarrow()),
            "lance_schema": _lance_schema_to_payload(merged_schema),
            "fragments": [updated_json],
            "created_files": created_files,
            "source_version": self.source_version,
            "source_fragment_id": fragment_id,
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
        self._add_columns_tables_by_shard.clear()
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
        self.mode = mode
        self.source_version = source_version
        self.planned_schema = planned_schema
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
    ) -> list[str]:
        if self.mode != "add_columns":
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
        if self.mode == "create" and existing is not None:
            raise ValueError(
                "Cannot create a Lance dataset at a location where one already exists."
            )
        if self.mode == "append":
            if existing is None:
                raise ValueError("Cannot append to a non-existent Lance dataset.")
            return
        if self.mode == "add_columns":
            self._validate_add_columns_fragment_coverage(lance, set())
            return
        if self.planned_schema is None:
            raise ValueError(
                f"Cannot {self.mode} an empty Lance dataset without a known schema"
            )
        operation = lance.LanceOperation.Overwrite(self.planned_schema, [])
        lance.LanceDataset.commit(
            self._dataset_uri(),
            operation,
            read_version=existing.version if existing is not None else 0,
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

        rejected_fragments: list[str] = []
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
                )
            except Exception as err:  # noqa: BLE001
                logger.warning(
                    "ignoring invalid rejected Lance metadata path={}: {}: {}",
                    rel_path,
                    type(err).__name__,
                    err,
                )
                continue
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
            self._commit_empty_output(_import_lance())
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

        self._validate_add_columns_fragment_coverage(lance, source_fragment_ids)

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
        else:
            operation = lance.LanceOperation.Overwrite(
                schema,
                [
                    lance.fragment.FragmentMetadata.from_json(fragment)
                    for fragment in fragment_json
                ],
            )
            read_version = existing.version if existing is not None else 0

        commit_options = (
            {"max_retries": 0} if self.mode in {"add_columns", "append"} else {}
        )
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
        for rel_path in sorted(set(cleanup_paths).union(metadata_paths)):
            try:
                self.output.rm(rel_path)
            except FileNotFoundError:
                continue
            except Exception as err:  # noqa: BLE001
                logger.warning(
                    "post-commit Lance metadata cleanup failed path={}: {}: {}",
                    rel_path,
                    type(err).__name__,
                    err,
                )

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
                except Exception as err:  # noqa: BLE001
                    logger.warning(
                        "Lance rejected-file cleanup failed path={}: {}: {}",
                        path,
                        type(err).__name__,
                        err,
                    )
            return
        for fragment in rejected_fragments:
            try:
                _remove_fragment_data(self.output, fragment)
            except Exception as err:  # noqa: BLE001
                logger.warning(
                    "Lance rejected-fragment cleanup failed: {}: {}",
                    type(err).__name__,
                    err,
                )


__all__ = ["LanceDatasetCommitReducerSink", "LanceDatasetSink", "LanceSink"]
