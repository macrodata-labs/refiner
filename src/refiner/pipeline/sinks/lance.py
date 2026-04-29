from __future__ import annotations

import base64
import concurrent.futures
import json
import posixpath
import queue as queue_module
import re
from typing import Any, Literal, get_args

import pyarrow as pa

from refiner.execution.asyncio.runtime import io_executor
from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.shard import SHARD_ID_COLUMN
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.reducer.file import (
    FileCleanupReducerSink,
    _compile_managed_path_pattern,
)
from refiner.worker.context import (
    get_active_stage_index,
    get_active_job_id,
    get_active_worker_token,
    get_finalized_workers,
)
from refiner.worker.metrics.api import log_throughput
from refiner.utils import check_required_dependencies

LanceWriteMode = Literal["create", "append", "overwrite"]
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
    ) -> None:
        _validate_write_mode(mode)
        self.output = DataFolder.resolve(output)
        self.mode = mode
        self._writers_by_shard: dict[str, _StreamingShardWriter] = {}
        self._schema_by_shard: dict[str, pa.Schema] = {}
        self._existing_schema: pa.Schema | None = None

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

    def write_shard_block(self, shard_id: str, block: Block) -> None:
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

    def close(self) -> None:
        first_error: Exception | None = None
        for writer in self._writers_by_shard.values():
            try:
                fragments = writer.finish()
                for fragment in fragments:
                    _remove_fragment_data(self.output, fragment)
            except Exception as err:  # noqa: BLE001
                if first_error is None:
                    first_error = err
        self._writers_by_shard.clear()
        self._schema_by_shard.clear()
        if first_error is not None:
            raise first_error

    def describe(self) -> tuple[str, str, dict[str, object]]:
        args: dict[str, object] = {
            "path": self.output.abs_path(),
            "mode": self.mode,
        }
        return ("write_lance_dataset", "writer", args)

    def build_reducer(self) -> BaseSink | None:
        return LanceDatasetCommitReducerSink(
            self.output,
            mode=self.mode,
        )


class LanceDatasetCommitReducerSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        mode: LanceWriteMode,
    ) -> None:
        _validate_write_mode(mode)
        self.output = DataFolder.resolve(output)
        self.mode = mode
        self._managed_path_pattern = _compile_managed_path_pattern(
            _METADATA_FILENAME_TEMPLATE
        )
        self._commit_ran = False

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
        return ("write_lance_dataset_commit", "writer", args)

    def write_shard_block(self, shard_id: str, block: Block) -> None:
        del shard_id, block
        self._run_commit()

    def _read_metadata(self, rel_path: str) -> tuple[pa.Schema, list[str]]:
        with self.output.open(rel_path, mode="rt", encoding="utf-8") as f:
            payload = json.load(f)
        schema_raw = payload.get("schema") if isinstance(payload, dict) else None
        fragment_raw = payload.get("fragments") if isinstance(payload, dict) else None
        if not isinstance(schema_raw, str) or not isinstance(fragment_raw, list):
            raise ValueError(f"Invalid Lance metadata payload: {rel_path}")
        return _schema_from_base64(schema_raw), [
            str(fragment) for fragment in fragment_raw
        ]

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
        for rel_path in rejected_paths:
            _, next_rejected_fragments = self._read_metadata(rel_path)
            rejected_fragments.extend(next_rejected_fragments)

        if not metadata_paths:
            for fragment in rejected_fragments:
                _remove_fragment_data(self.output, fragment)
            for rel_path in cleanup_paths:
                try:
                    self.output.rm(rel_path)
                except FileNotFoundError:
                    continue
            return

        lance = _import_lance()
        fragment_json: list[str] = []
        schema: pa.Schema | None = None
        for rel_path in sorted(metadata_paths):
            next_schema, next_fragments = self._read_metadata(rel_path)
            if schema is None:
                schema = next_schema
            elif not schema.equals(next_schema):
                raise ValueError(
                    "Cannot commit Lance fragments with inconsistent schemas."
                )
            fragment_json.extend(next_fragments)

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
        else:
            operation = lance.LanceOperation.Overwrite(
                schema,
                [
                    lance.fragment.FragmentMetadata.from_json(fragment)
                    for fragment in fragment_json
                ],
            )
            read_version = existing.version if existing is not None else 0

        lance.LanceDataset.commit(
            self._dataset_uri(),
            operation,
            read_version=read_version,
        )
        for fragment in rejected_fragments:
            _remove_fragment_data(self.output, fragment)
        for rel_path in cleanup_paths:
            try:
                self.output.rm(rel_path)
            except FileNotFoundError:
                continue


__all__ = ["LanceDatasetCommitReducerSink", "LanceDatasetSink", "LanceSink"]
