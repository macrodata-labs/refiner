from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.shard import SHARD_ID_COLUMN
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sinks.assets import AssetUploadManager, MissingAssetPolicy
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.reducer.file import FileCleanupReducerSink
from refiner.worker.context import get_active_worker_token
from refiner.worker.metrics.api import log_throughput


class ParquetSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}__w{worker_id}.parquet",
        compression: str | None = None,
        upload_assets: bool = False,
        assets_subdir: str = "assets",
        max_asset_uploads_in_flight: int = 16,
        missing_asset_policy: MissingAssetPolicy = "error",
    ):
        self.output = DataFolder.resolve(output)
        self.filename_template = filename_template
        self.compression = compression
        self.upload_assets = upload_assets
        self.assets_subdir = assets_subdir
        self.missing_asset_policy = missing_asset_policy
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._schema: pa.Schema | None = None
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
        self._schema = schema
        if self._assets is not None:
            self._assets.set_input_schema(schema)

    def _relpath(self, shard_id: str) -> str:
        return self.filename_template.format(
            shard_id=shard_id,
            worker_id=get_active_worker_token(),
        )

    def _writer(self, shard_id: str, schema: pa.Schema) -> pq.ParquetWriter:
        writer = self._writers.get(shard_id)
        if writer is not None:
            return writer
        sink = self.output.open(self._relpath(shard_id), mode="wb")
        writer = pq.ParquetWriter(sink, schema, compression=self.compression)
        self._writers[shard_id] = writer
        return writer

    def write_shard_block(self, shard_id: str, block: Block) -> None:
        if isinstance(block, Tabular):
            table = block.table
        else:
            if self._assets is not None:
                self._assets.require_input_schema()
            table = (
                Tabular.from_rows(block, schema=self._schema).table
                if not block
                else block[0].tabular_type.from_rows(block, schema=self._schema).table
            )
        if SHARD_ID_COLUMN in table.schema.names:
            table = table.drop_columns([SHARD_ID_COLUMN])
        if self._assets is None:
            self._writer(shard_id, table.schema).write_table(table)
            return
        self._writer(
            shard_id,
            (table := self._assets.rewrite_table(shard_id, table)).schema,
        ).write_table(table)

    def on_shard_complete(self, shard_id: str) -> None:
        writer = self._writers.pop(shard_id, None)
        if writer is not None:
            writer.close()
            log_throughput("files_written", 1, shard_id=shard_id, unit="files")

    def close(self) -> None:
        try:
            if self._assets is not None:
                self._assets.flush()
        finally:
            for writer in self._writers.values():
                writer.close()
            self._writers.clear()

    def describe(self) -> tuple[str, str, dict[str, object]]:
        args: dict[str, object] = {
            "path": self.output.abs_path(),
            "filename_template": self.filename_template,
        }
        if self.compression is not None:
            args["compression"] = self.compression
        if self.upload_assets:
            args["upload_assets"] = True
            args["assets_subdir"] = self.assets_subdir
            args["missing_asset_policy"] = self.missing_asset_policy
        return ("write_parquet", "writer", args)

    def build_reducer(self) -> BaseSink | None:
        return FileCleanupReducerSink(
            output=self.output,
            filename_template=self.filename_template,
            reducer_name="write_parquet_reduce",
            assets_subdir=self.assets_subdir if self.upload_assets else None,
        )


__all__ = ["ParquetSink"]
