from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.datatype import (
    DTypeMapping,
    apply_dtypes_to_table,
    dtype_to_plan,
    schema_with_dtypes,
)
from refiner.pipeline.data.block import strip_internal_columns
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


class ParquetSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}__w{worker_id}.parquet",
        compression: str | None = None,
        assets: AssetWriteConfig | None = None,
        dtypes: DTypeMapping | None = None,
    ):
        self.output = DataFolder.resolve(output)
        self.filename_template = filename_template
        self.compression = compression
        self.assets = assets
        self.dtypes = dtypes
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._asset_rows_written: dict[str, int] = {}
        self._schema: pa.Schema | None = None
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
        self._schema = schema_with_dtypes(schema, self.dtypes, preserve_metadata=False)
        if self._assets is not None:
            self._assets.set_input_schema(self._schema)

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

    def write_shard_block(self, shard_id: str, block: Block) -> int:
        if isinstance(block, Tabular):
            table = apply_dtypes_to_table(
                block.table,
                self.dtypes,
                preserve_metadata=False,
            )
        else:
            if self._assets is not None:
                self._assets.require_input_schema()
            table = (
                Tabular.from_rows(block, schema=self._schema).table
                if not block
                else block[0].tabular_type.from_rows(block, schema=self._schema).table
            )
        table = strip_internal_columns(table)
        if self._assets is None:
            self._writer(shard_id, table.schema).write_table(table)
            return table.num_rows
        if isinstance(self._assets, AssetUploadManager):
            self._write_ready_asset_blocks(self._assets.submit_table(shard_id, table))
            return self._asset_rows_written.pop(shard_id, 0)
        self._writer(
            shard_id,
            (table := self._assets.rewrite_table(shard_id, table)).schema,
        ).write_table(table)
        return table.num_rows

    def _write_ready_asset_blocks(self, blocks: list[ReadyAssetBlock]) -> None:
        for ready in blocks:
            assert isinstance(ready.block, pa.Table)
            self._writer(ready.shard_id, ready.block.schema).write_table(ready.block)
            self._asset_rows_written[ready.shard_id] = (
                self._asset_rows_written.get(ready.shard_id, 0) + ready.block.num_rows
            )

    def on_shard_complete(self, shard_id: str) -> int | None:
        if isinstance(self._assets, AssetUploadManager):
            self._write_ready_asset_blocks(self._assets.on_shard_complete(shard_id))
        elif self._assets is not None:
            self._assets.on_shard_complete(shard_id)
        writer = self._writers.pop(shard_id, None)
        if writer is not None:
            writer.close()
            log_throughput("files_written", 1, shard_id=shard_id, unit="files")
        if isinstance(self._assets, AssetUploadManager):
            return self._asset_rows_written.pop(shard_id, 0)
        return None

    def close(self) -> None:
        try:
            if self._assets is not None:
                self._assets.close()
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
        if self.assets is not None:
            args["assets"] = asset_config_to_plan(self.assets)
        if self.dtypes:
            args["dtypes"] = {
                key: dtype_to_plan(dtype) for key, dtype in self.dtypes.items()
            }
        return ("write_parquet", "writer", args)

    def build_reducer(self) -> BaseSink | None:
        return FileCleanupReducerSink(
            output=self.output,
            filename_template=self.filename_template,
            reducer_name="write_parquet_reduce",
            assets_subdir=self.assets.subdir if self.assets is not None else None,
        )


__all__ = ["ParquetSink"]
