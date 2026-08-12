from __future__ import annotations

import posixpath
from pathlib import PureWindowsPath
from typing import Any

import pyarrow as pa

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.assets import (
    AssetUploadManager,
    AssetWriteConfig,
    BlobAssetConfig,
    BlobAssetManager,
    FileAssetConfig,
    asset_config_to_plan,
)
from refiner.pipeline.sinks.lance_utils import block_to_table, validate_lance_uri
from refiner.pipeline.sinks.reducer.file import FileCleanupReducerSink
from refiner.utils import check_required_dependencies
from refiner.worker.context import get_active_worker_token
from refiner.worker.metrics.api import log_throughput


def _import_lance_file_writer() -> Any:
    check_required_dependencies("write_lance", [("lance", "pylance")], dist="lance")
    from lance.file import LanceFileWriter

    return LanceFileWriter


class LanceSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}__w{worker_id}.lance",
        assets: AssetWriteConfig | None = None,
    ) -> None:
        self.output = DataFolder.resolve(output)
        if self.output.has_explicit_filesystem_configuration:
            raise ValueError(
                "write_lance does not support configured fsspec handles; pass a URI "
                "whose credentials and endpoint are available to Lance"
            )
        validate_lance_uri(self.output.abs_path())
        normalized_template = posixpath.normpath(filename_template)
        if (
            not filename_template
            or normalized_template != filename_template
            or normalized_template in {".", ".."}
            or normalized_template.startswith("../")
            or normalized_template.startswith("/")
            or "\\" in filename_template
            or "://" in filename_template
            or PureWindowsPath(filename_template).drive
        ):
            raise ValueError("filename_template must be a normalized relative path")
        self.filename_template = filename_template
        self.assets = assets
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
        self._writers: dict[str, Any] = {}

    def set_input_schema(self, schema: pa.Schema | None) -> None:
        if self._assets is not None:
            self._assets.set_input_schema(schema)

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
        if self._assets is not None:
            table = self._assets.rewrite_table(shard_id, table)
        self._writer(shard_id, table.schema).write_batch(table)

    def on_shard_complete(self, shard_id: str) -> None:
        if self._assets is not None:
            self._assets.on_shard_complete(shard_id)
        writer = self._writers.pop(shard_id, None)
        if writer is not None:
            writer.close()
            log_throughput("files_written", 1, shard_id=shard_id, unit="files")

    def close(self) -> None:
        first_error: Exception | None = None
        if self._assets is not None:
            self._assets.close()
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
        args: dict[str, object] = {
            "path": self.output.abs_path(),
            "filename_template": self.filename_template,
        }
        if self.assets is not None:
            args["assets"] = asset_config_to_plan(self.assets)
        return ("write_lance", "writer", args)

    def build_reducer(self) -> BaseSink | None:
        return FileCleanupReducerSink(
            output=self.output,
            filename_template=self.filename_template,
            reducer_name="write_lance_reduce",
            assets_subdir=self.assets.subdir if self.assets is not None else None,
        )
