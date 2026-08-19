from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, IO, cast

import numpy as np
import pyarrow as pa

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block, strip_internal_columns
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sinks.assets import (
    AssetUploadManager,
    AssetWriteConfig,
    BlobAssetConfig,
    BlobAssetManager,
    FileAssetConfig,
    ReadyAssetBlock,
    asset_config_to_plan,
)
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.reducer.file import FileCleanupReducerSink
from refiner.worker.context import get_active_worker_token
from refiner.worker.metrics.api import log_throughput


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return cast(Any, obj).tolist()
    if isinstance(obj, np.generic):
        return cast(Any, obj).item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class JsonlSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}__w{worker_id}.jsonl",
        assets: AssetWriteConfig | None = None,
    ):
        self.output = DataFolder.resolve(output)
        self.filename_template = filename_template
        self.assets = assets
        self._files: dict[str, IO[str]] = {}
        self._asset_rows_written: dict[str, int] = {}
        self._encoder = json.JSONEncoder(
            ensure_ascii=True,
            separators=(",", ":"),
            default=_json_default,
        )
        if isinstance(assets, FileAssetConfig):
            self._assets = AssetUploadManager(
                self.output,
                assets_subdir=assets.subdir,
                filename_template=filename_template,
                max_uploads_in_flight=assets.max_in_flight,
                missing_asset_policy=assets.missing_policy,
            )
        elif isinstance(assets, BlobAssetConfig):
            self._assets = BlobAssetManager(
                self.output,
                config=assets,
                filename_template=filename_template,
            )
        else:
            self._assets = None

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

    def _write_rows(self, shard_id: str, rows: Iterable[Mapping[str, object]]) -> int:
        file = self._file(shard_id)
        count = 0
        for row in rows:
            file.write(
                self._encoder.encode(row.to_dict() if isinstance(row, Row) else row)
            )
            file.write("\n")
            count += 1
        return count

    def _write_table_rows(self, shard_id: str, table: pa.Table) -> int:
        if self._assets is not None and not isinstance(
            self._assets, AssetUploadManager
        ):
            return self._write_rows(
                shard_id, self._assets.rewrite_table(shard_id, table).to_pylist()
            )
        count = 0
        for batch in table.to_batches(max_chunksize=4096):
            count += self._write_rows(shard_id, batch.to_pylist())
        return count

    def write_shard_block(self, shard_id: str, block: Block) -> int:
        if isinstance(self._assets, AssetUploadManager):
            if isinstance(block, Tabular):
                ready = self._assets.submit_table(
                    shard_id, strip_internal_columns(block.table)
                )
            else:
                ready = self._assets.submit_rows(shard_id, block)
            self._write_ready_asset_blocks(ready)
            return self._asset_rows_written.pop(shard_id, 0)
        if isinstance(block, Tabular):
            return self._write_table_rows(
                shard_id,
                strip_internal_columns(block.table),
            )
        rows = block
        if self._assets is not None:
            rows = self._assets.rewrite_rows(shard_id, rows)
        return self._write_rows(shard_id, rows)

    def _write_ready_asset_blocks(self, blocks: list[ReadyAssetBlock]) -> None:
        for ready in blocks:
            rows = (
                ready.block.to_pylist()
                if isinstance(ready.block, pa.Table)
                else ready.block
            )
            count = self._write_rows(ready.shard_id, rows)
            self._asset_rows_written[ready.shard_id] = (
                self._asset_rows_written.get(ready.shard_id, 0) + count
            )

    def on_shard_complete(self, shard_id: str) -> int | None:
        if isinstance(self._assets, AssetUploadManager):
            self._write_ready_asset_blocks(self._assets.on_shard_complete(shard_id))
        elif self._assets is not None:
            self._assets.on_shard_complete(shard_id)
        file = self._files.pop(shard_id, None)
        if file is not None:
            file.close()
            log_throughput("files_written", 1, shard_id=shard_id, unit="files")
        if isinstance(self._assets, AssetUploadManager):
            return self._asset_rows_written.pop(shard_id, 0)
        return None

    def close(self) -> None:
        try:
            if self._assets is not None:
                self._assets.close()
        finally:
            for file in self._files.values():
                file.close()
            self._files.clear()

    def describe(self) -> tuple[str, str, dict[str, object]]:
        args: dict[str, object] = {
            "path": self.output.abs_path(),
            "filename_template": self.filename_template,
        }
        if self.assets is not None:
            args["assets"] = asset_config_to_plan(self.assets)
        return ("write_jsonl", "writer", args)

    def build_reducer(self) -> BaseSink | None:
        return FileCleanupReducerSink(
            output=self.output,
            filename_template=self.filename_template,
            reducer_name="write_jsonl_reduce",
            assets_subdir=self.assets.subdir if self.assets is not None else None,
        )


__all__ = ["JsonlSink"]
