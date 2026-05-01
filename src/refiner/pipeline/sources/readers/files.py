from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from typing import Any

from fsspec import AbstractFileSystem

from refiner.execution.asyncio.runtime import io_executor
from refiner.execution.asyncio.window import AsyncWindow
from refiner.io import DataFile
from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.datatype import DTypeMapping
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.shard import FilePartsDescriptor
from refiner.pipeline.sources.readers.base import BaseReader, Shard, SourceUnit
from refiner.pipeline.sources.readers.utils import DEFAULT_TARGET_SHARD_BYTES


class FilesReader(BaseReader):
    """Generic file listing/content reader.

    Emits one row per resolved input file. File contents are opened only when
    `content_column` is set.
    """

    def __init__(
        self,
        inputs: DataFileSetLike,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        recursive: bool = False,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
        file_path_column: str | None = "file_path",
        content_column: str | None = None,
        max_in_flight: int = 8,
        dtypes: DTypeMapping | None = None,
    ):
        if file_path_column is None and content_column is None:
            raise ValueError("read_files requires file_path_column or content_column")
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be > 0")
        if (
            file_path_column is not None
            and content_column is not None
            and file_path_column == content_column
        ):
            raise ValueError("file_path_column and content_column must differ")
        super().__init__(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            file_path_column=file_path_column,
            split_by_bytes=False,
            dtypes=dtypes,
        )
        self.content_column = content_column
        self.max_in_flight = int(max_in_flight)

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out["content_column"] = self.content_column
        out["max_in_flight"] = self.max_in_flight
        return out

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)

        if self.content_column is None:
            for part in descriptor.parts:
                source = self.fileset.resolve_file(part.source_index, part.path)
                yield DictRow(self._path_row(source))
            return

        if self.max_in_flight == 1:
            for part in descriptor.parts:
                source = self.fileset.resolve_file(part.source_index, part.path)
                yield DictRow(self._content_row(source))
            return

        window = AsyncWindow[dict[str, Any]](
            max_in_flight=self.max_in_flight,
            preserve_order=True,
        )
        for part in descriptor.parts:
            source = self.fileset.resolve_file(part.source_index, part.path)
            window.submit_blocking(self._content_row_async(source))
            for row in window.take_completed():
                yield DictRow(row)
        for row in window.drain():
            yield DictRow(row)

    def _path_row(self, source: DataFile) -> dict[str, Any]:
        if self.file_path_column is None:
            return {}
        return {self.file_path_column: source.abs_path()}

    def _content_row(self, source: DataFile) -> dict[str, Any]:
        content_column = self.content_column
        assert content_column is not None
        row = self._path_row(source)
        with source.open(mode="rb") as raw:
            row[content_column] = raw.read()
        return row

    async def _content_row_async(self, source: DataFile) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(io_executor(), self._content_row, source)


__all__ = ["FilesReader"]
