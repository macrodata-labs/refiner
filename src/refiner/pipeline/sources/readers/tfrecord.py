from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from fsspec import AbstractFileSystem
import pyarrow as pa

from refiner.io import DataFile
from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.shard import FilePartsDescriptor
from refiner.pipeline.data.tabular import Tabular, set_or_append_column
from refiner.pipeline.sources.readers._tensorflow import (
    require_tensorflow,
    tensorflow_batch_to_table,
)
from refiner.pipeline.sources.readers.base import BaseReader, Shard, SourceUnit
from refiner.pipeline.sources.readers.utils import DEFAULT_TARGET_SHARD_BYTES


class TfrecordReader(BaseReader):
    """TFRecord reader backed by TensorFlow parsing.

    Sharding is file-level: small files may be grouped into one shard, but a
    single TFRecord file is never split internally.
    """

    name = "read_tfrecords"

    def __init__(
        self,
        inputs: DataFileSetLike,
        *,
        features: Mapping[str, Any],
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        recursive: bool = False,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
        batch_size: int = 1024,
        compression: str | None = "auto",
        num_parallel_calls: int | None = None,
        prefetch: int | None = 1,
        file_path_column: str | None = "file_path",
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        super().__init__(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            extensions=(
                ".tfrecord",
                ".tfrecords",
                ".tfrec",
                ".tfrecord.gz",
                ".tfrecords.gz",
                ".tfrec.gz",
                ".tfrecord.zlib",
                ".tfrecords.zlib",
                ".tfrec.zlib",
            ),
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            file_path_column=file_path_column,
            split_by_bytes=False,
        )
        self.tf = require_tensorflow()
        self.features = dict(features)
        self.batch_size = int(batch_size)
        if compression not in {None, "auto", "gzip", "zlib"}:
            raise ValueError("compression must be one of: None, 'auto', 'gzip', 'zlib'")
        self.compression = compression
        self.num_parallel_calls = num_parallel_calls
        self.prefetch = prefetch
        self._add_file_path = (
            self.file_path_column is not None
            and self.file_path_column not in self.features
        )

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "batch_size": self.batch_size,
                "compression": self.compression,
                "features": sorted(self.features),
            }
        )
        return description

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)
        dataset = None
        for part in descriptor.parts:
            source = self.fileset.resolve_file(part.source_index, part.path)
            if not source.is_local:
                raise ValueError(
                    "TfrecordReader reads with TensorFlow and only supports local "
                    "paths. Custom fsspec filesystems are not supported."
                )
            path = source.abs_path()
            records = self.tf.data.TFRecordDataset(
                [path],
                compression_type=self._compression_type(source),
            )
            if self._add_file_path:
                paths = self.tf.data.Dataset.from_tensors(path).repeat()
                records = self.tf.data.Dataset.zip(
                    (
                        records,
                        paths,
                    )
                )
            dataset = records if dataset is None else dataset.concatenate(records)
        if dataset is None:
            return

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(
            self._parse_batch,
            num_parallel_calls=self.num_parallel_calls,
        )
        if self.prefetch is not None:
            dataset = dataset.prefetch(self.prefetch)

        for batch in dataset:
            table = tensorflow_batch_to_table(batch)
            if self._add_file_path:
                assert self.file_path_column is not None
                table = set_or_append_column(
                    table,
                    self.file_path_column,
                    table.column(self.file_path_column).cast(pa.string()),
                )
            if table.num_rows > 0:
                yield Tabular(table)

    def _parse_batch(self, records, paths=None):
        parsed = self.tf.io.parse_example(records, self.features)
        if self._add_file_path:
            assert self.file_path_column is not None
            parsed[self.file_path_column] = paths
        return parsed

    def _compression_type(self, source: DataFile) -> str:
        if self.compression is None:
            return ""
        if self.compression == "gzip":
            return "GZIP"
        if self.compression == "zlib":
            return "ZLIB"
        path = source.path.lower()
        if path.endswith(".gz"):
            return "GZIP"
        if path.endswith(".zlib"):
            return "ZLIB"
        return ""


__all__ = ["TfrecordReader"]
