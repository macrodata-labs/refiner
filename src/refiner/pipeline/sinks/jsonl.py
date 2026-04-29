from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import IO

import pyarrow as pa

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sinks.assets import AssetUploadManager, MissingAssetPolicy
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.reducer.file import FileCleanupReducerSink
from refiner.worker.context import get_active_worker_token
from refiner.worker.metrics.api import log_throughput


class JsonlSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}__w{worker_id}.jsonl",
        upload_assets: bool = False,
        assets_subdir: str = "assets",
        max_asset_uploads_in_flight: int = 16,
        missing_asset_policy: MissingAssetPolicy = "error",
    ):
        self.output = DataFolder.resolve(output)
        self.filename_template = filename_template
        self.upload_assets = upload_assets
        self.assets_subdir = assets_subdir
        self.missing_asset_policy = missing_asset_policy
        self._files: dict[str, IO[str]] = {}
        self._encoder = json.JSONEncoder(ensure_ascii=True, separators=(",", ":"))
        self._assets = (
            AssetUploadManager(
                self.output,
                assets_subdir=assets_subdir,
                filename_template=filename_template,
                max_uploads_in_flight=max_asset_uploads_in_flight,
                missing_asset_policy=missing_asset_policy,
            )
            if upload_assets
            else None
        )
        if self._assets is not None:
            self.assets_subdir = self._assets.assets_subdir

    def set_input_schema(self, schema: pa.Schema | None) -> None:
        if self._assets is not None:
            self._assets.set_input_schema(schema)

    def _relpath(self, shard_id: str) -> str:
        return self.filename_template.format(
            shard_id=shard_id,
            worker_id=get_active_worker_token(),
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
            file.write(
                self._encoder.encode(row.to_dict() if isinstance(row, Row) else row)
            )
            file.write("\n")

    def _write_table_rows(self, shard_id: str, table: pa.Table) -> None:
        if self._assets is not None:
            self._write_rows(
                shard_id, self._assets.rewrite_table(shard_id, table).to_pylist()
            )
            return
        for batch in table.to_batches(max_chunksize=4096):
            self._write_rows(shard_id, batch.to_pylist())

    def write_shard_block(self, shard_id: str, block: Block) -> None:
        if isinstance(block, Tabular):
            self._write_table_rows(shard_id, block.table)
        else:
            rows = block
            if self._assets is not None:
                rows = self._assets.rewrite_rows(shard_id, rows)
            self._write_rows(shard_id, rows)

    def on_shard_complete(self, shard_id: str) -> None:
        file = self._files.pop(shard_id, None)
        if file is not None:
            file.close()
            log_throughput("files_written", 1, shard_id=shard_id, unit="files")

    def close(self) -> None:
        try:
            if self._assets is not None:
                self._assets.flush()
        finally:
            for file in self._files.values():
                file.close()
            self._files.clear()

    def describe(self) -> tuple[str, str, dict[str, object]]:
        args: dict[str, object] = {
            "path": self.output.abs_path(),
            "filename_template": self.filename_template,
        }
        if self.upload_assets:
            args["upload_assets"] = True
            args["assets_subdir"] = self.assets_subdir
            args["missing_asset_policy"] = self.missing_asset_policy
        return ("write_jsonl", "writer", args)

    def build_reducer(self) -> BaseSink | None:
        return FileCleanupReducerSink(
            output=self.output,
            filename_template=self.filename_template,
            reducer_name="write_jsonl_reduce",
            assets_subdir=self.assets_subdir if self.upload_assets else None,
        )


__all__ = ["JsonlSink"]
