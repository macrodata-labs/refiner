from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Mapping
from typing import Any, cast

from fsspec import AbstractFileSystem

from refiner.execution.asyncio.runtime import io_executor
from refiner.execution.asyncio.window import AsyncWindow
from refiner.io import DataFile
from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.datatype import DTypeMapping
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.shard import FilePart, FilePartsDescriptor
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
        size_column: str | None = "size",
        decode_fn: Callable[[bytes], Any] | None = None,
        max_in_flight: int = 8,
        dtypes: DTypeMapping | None = None,
    ):
        columns = [
            name
            for name in (file_path_column, content_column, size_column)
            if name is not None
        ]
        if not columns:
            raise ValueError(
                "read_files requires file_path_column, content_column, or size_column"
            )
        if len(columns) != len(set(columns)):
            raise ValueError(
                "file_path_column, content_column, and size_column must differ"
            )
        if decode_fn is not None and content_column is None:
            raise ValueError("decode_fn requires content_column")
        if content_column is not None and max_in_flight <= 0:
            raise ValueError("max_in_flight must be > 0")
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
        self.size_column = size_column
        self.decode_fn = decode_fn
        self.max_in_flight = int(max_in_flight)

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out["content_column"] = self.content_column
        out["size_column"] = self.size_column
        out["decode_fn"] = self.decode_fn is not None
        out["max_in_flight"] = self.max_in_flight
        return out

    def list_shards(self) -> list[Shard]:
        return [
            Shard.from_file_parts(
                tuple(
                    FilePart(
                        path=part.path,
                        start=part.start,
                        end=part.end,
                        source_index=part.source_index,
                        # Base planning already populated DataFileSet's size cache.
                        metadata={
                            **part.metadata,
                            "size": self.fileset.size(part.source_index, part.path),
                        },
                    )
                    for part in cast(FilePartsDescriptor, shard.descriptor).parts
                ),
                global_ordinal=shard.global_ordinal,
                start_key=shard.start_key,
                end_key=shard.end_key,
            )
            for shard in super().list_shards()
        ]

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)

        if self.content_column is None or self.max_in_flight == 1:
            for part in descriptor.parts:
                source = self.fileset.resolve_file(part.source_index, part.path)
                row = (
                    self._base_row(source, part)
                    if self.content_column is None
                    else self._content_row(source, part)
                )
                yield DictRow(row)
            return

        window: AsyncWindow[dict[str, Any]] = AsyncWindow(
            max_in_flight=self.max_in_flight,
            preserve_order=True,
        )
        try:
            for part in descriptor.parts:
                source = self.fileset.resolve_file(part.source_index, part.path)
                window.submit_blocking(self._content_row_async(source, part))
                for row in window.take_completed():
                    yield DictRow(row)
            for row in window.drain():
                yield DictRow(row)
        finally:
            window.cancel_pending()

    def _base_row(self, source: DataFile, part: FilePart) -> dict[str, Any]:
        row: dict[str, Any] = {}
        if self.file_path_column is not None:
            row[self.file_path_column] = source.abs_path()
        if self.size_column is not None:
            row[self.size_column] = part.metadata.get("size")
        return row

    def _content_row(self, source: DataFile, part: FilePart) -> dict[str, Any]:
        content_column = self.content_column
        assert content_column is not None
        row = self._base_row(source, part)
        with source.open(mode="rb") as raw:
            data = raw.read()
        row[content_column] = (
            self.decode_fn(data) if self.decode_fn is not None else data
        )
        return row

    async def _content_row_async(
        self, source: DataFile, part: FilePart
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            io_executor(), self._content_row, source, part
        )


__all__ = ["FilesReader"]
