from __future__ import annotations

import ntpath
from typing import Any

import pyarrow as pa

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.lance_utils import block_to_table, validate_lance_uri
from refiner.pipeline.sinks.reducer.file import (
    FileCleanupReducerSink,
    _compile_output_path_patterns,
)
from refiner.utils import check_required_dependencies
from refiner.worker.context import get_active_worker_token
from refiner.worker.metrics.api import log_throughput


def _import_lance_file_writer() -> Any:
    check_required_dependencies("write_lance", [("lance", "pylance")], dist="lance")
    from lance.file import LanceFileWriter

    return LanceFileWriter


def _validate_filename_template(filename_template: str) -> None:
    drive, _ = ntpath.splitdrive(filename_template)
    if (
        not filename_template
        or drive
        or filename_template.startswith("/")
        or "\\" in filename_template
        or "://" in filename_template
        or any(part in {"", ".", ".."} for part in filename_template.split("/"))
    ):
        raise ValueError("filename_template must be a normalized relative path")
    _compile_output_path_patterns(filename_template)


class LanceSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}__w{worker_id}.lance",
    ) -> None:
        _validate_filename_template(filename_template)
        self.output = DataFolder.resolve(output)
        if self.output.has_explicit_filesystem_configuration:
            raise ValueError(
                "write_lance does not support configured fsspec handles; pass a URI "
                "whose credentials and endpoint are available to Lance"
            )
        validate_lance_uri(self.output.abs_path())
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
        lance_file_writer = _import_lance_file_writer()
        writer = lance_file_writer(
            self.output.abs_path(self._relpath(shard_id)), schema
        )
        self._writers[shard_id] = writer
        return writer

    def write_shard_block(self, shard_id: str, block: Block) -> None:
        table = block_to_table(block)
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
