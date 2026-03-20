from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import IO

import pyarrow as pa

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sinks.base import BaseSink
from refiner.worker.context import get_active_run_handle
from refiner.worker.metrics.api import log_throughput


class JsonlSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}__w{worker_id}.jsonl",
    ):
        self.output = DataFolder.resolve(output)
        self.filename_template = filename_template
        self._files: dict[str, IO[str]] = {}
        self._encoder = json.JSONEncoder(ensure_ascii=True, separators=(",", ":"))

    def _relpath(self, shard_id: str) -> str:
        return self.filename_template.format(
            shard_id=shard_id,
            worker_id=get_active_run_handle().worker_token,
        )

    def _file(self, shard_id: str) -> IO[str]:
        file = self._files.get(shard_id)
        if file is not None:
            return file
        file = self.output.open(self._relpath(shard_id), mode="wt", encoding="utf-8")
        self._files[shard_id] = file
        return file

    def _write_rows(self, shard_id: str, rows: Iterable[Mapping[str, object]]) -> None:
        file = self._file(shard_id)
        for row in rows:
            file.write(self._encoder.encode(row))
            file.write("\n")

    def _write_table_rows(self, shard_id: str, table: pa.Table) -> None:
        for batch in table.to_batches(max_chunksize=4096):
            self._write_rows(shard_id, batch.to_pylist())

    def write_shard_block(self, shard_id: str, block: Block) -> None:
        if isinstance(block, Tabular):
            self._write_table_rows(shard_id, block.table)
        else:
            self._write_rows(shard_id, (row.to_dict() for row in block))

    def on_shard_complete(self, shard_id: str) -> None:
        file = self._files.pop(shard_id, None)
        if file is not None:
            file.close()
            log_throughput("files_written", 1, shard_id=shard_id, unit="files")

    def close(self) -> None:
        for file in self._files.values():
            file.close()
        self._files.clear()

    def describe(self) -> tuple[str, str, dict[str, object]]:
        return (
            "write_jsonl",
            "writer",
            {
                "path": self.output.abs_path(),
                "filename_template": self.filename_template,
            },
        )


__all__ = ["JsonlSink"]
