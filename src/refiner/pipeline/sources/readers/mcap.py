from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from fsspec import AbstractFileSystem

from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.shard import FilePartsDescriptor
from refiner.pipeline.sources.readers.base import BaseReader, Shard, SourceUnit
from refiner.pipeline.sources.readers.utils import DEFAULT_TARGET_SHARD_BYTES
from refiner.utils import check_required_dependencies


class McapReader(BaseReader):
    """Read MCAP files as raw message rows.

    The reader intentionally does not synchronize topics into robotics episodes.
    It exposes logged messages; downstream code can choose a topic alignment policy.
    """

    name = "read_mcap"

    def __init__(
        self,
        inputs: DataFileSetLike,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        recursive: bool = False,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
        topics: Sequence[str] | None = None,
        file_path_column: str | None = "file_path",
        data_column: str = "data",
    ):
        super().__init__(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            extensions=(".mcap",),
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            file_path_column=file_path_column,
            split_by_bytes=False,
        )
        self.topics = tuple(topics) if topics is not None else None
        self.data_column = data_column

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "topics": list(self.topics) if self.topics is not None else None,
                "data_column": self.data_column,
            }
        )
        return description

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)
        check_required_dependencies("read_mcap", ["mcap"], dist="mcap")
        from mcap.reader import make_reader

        for part in descriptor.parts:
            source = self.fileset.resolve_file(part.source_index, part.path)
            with source.open(mode="rb") as stream:
                reader = make_reader(stream)
                for schema, channel, message in reader.iter_messages(
                    topics=self.topics
                ):
                    row = {
                        "topic": channel.topic,
                        "log_time": int(message.log_time),
                        "publish_time": int(message.publish_time),
                        "sequence": int(message.sequence),
                        "message_encoding": channel.message_encoding,
                        "schema_id": int(channel.schema_id),
                        "schema_name": schema.name if schema is not None else None,
                        "schema_encoding": schema.encoding
                        if schema is not None
                        else None,
                        "schema_data": bytes(schema.data)
                        if schema is not None
                        else None,
                        self.data_column: bytes(message.data),
                    }
                    yield DictRow(self._with_file_path(row, source))


__all__ = ["McapReader"]
