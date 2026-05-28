from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from fsspec import AbstractFileSystem
import pyarrow as pa

from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.data.shard import FilePartsDescriptor
from refiner.pipeline.sources.readers.base import BaseReader, Shard, SourceUnit
from refiner.pipeline.sources.readers.utils import DEFAULT_TARGET_SHARD_BYTES
from refiner.utils import check_required_dependencies


class McapReader(BaseReader):
    """Read each MCAP file as one row with a nested message table.

    The reader intentionally does not synchronize topics into robotics episodes.
    It exposes the file's logged messages as Arrow data; downstream code can
    choose a topic alignment policy.
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
        messages_column: str = "messages",
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
        self.messages_column = messages_column
        if data_column in {
            "topic",
            "log_time",
            "publish_time",
            "sequence",
            "message_encoding",
            "schema_id",
            "schema_name",
            "schema_encoding",
            "schema_data",
        }:
            raise ValueError(
                f"data_column conflicts with MCAP metadata column: {data_column!r}"
            )
        self.data_column = data_column

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "topics": list(self.topics) if self.topics is not None else None,
                "messages_column": self.messages_column,
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
            columns: dict[str, list[Any]] = {
                "topic": [],
                "log_time": [],
                "publish_time": [],
                "sequence": [],
                "message_encoding": [],
                "schema_id": [],
                "schema_name": [],
                "schema_encoding": [],
                "schema_data": [],
                self.data_column: [],
            }
            with source.open(mode="rb") as stream:
                reader = make_reader(stream)
                for schema, channel, message in reader.iter_messages(
                    topics=self.topics
                ):
                    columns["topic"].append(channel.topic)
                    columns["log_time"].append(int(message.log_time))
                    columns["publish_time"].append(int(message.publish_time))
                    columns["sequence"].append(int(message.sequence))
                    columns["message_encoding"].append(channel.message_encoding)
                    columns["schema_id"].append(int(channel.schema_id))
                    columns["schema_name"].append(
                        schema.name if schema is not None else None
                    )
                    columns["schema_encoding"].append(
                        schema.encoding if schema is not None else None
                    )
                    columns["schema_data"].append(
                        bytes(schema.data) if schema is not None else None
                    )
                    columns[self.data_column].append(bytes(message.data))
            row = {
                self.messages_column: Tabular(
                    _messages_table(columns, self.data_column)
                ),
                "message_count": len(columns["topic"]),
                "topics": sorted(set(columns["topic"])),
            }
            yield DictRow(self._with_file_path(row, source))


def _messages_table(columns: dict[str, list[Any]], data_column: str) -> pa.Table:
    return pa.table(
        {
            "topic": pa.array(columns["topic"], type=pa.string()),
            "log_time": pa.array(columns["log_time"], type=pa.int64()),
            "publish_time": pa.array(columns["publish_time"], type=pa.int64()),
            "sequence": pa.array(columns["sequence"], type=pa.int64()),
            "message_encoding": pa.array(columns["message_encoding"], type=pa.string()),
            "schema_id": pa.array(columns["schema_id"], type=pa.int64()),
            "schema_name": pa.array(columns["schema_name"], type=pa.string()),
            "schema_encoding": pa.array(columns["schema_encoding"], type=pa.string()),
            "schema_data": pa.array(columns["schema_data"], type=pa.binary()),
            data_column: pa.array(columns[data_column], type=pa.binary()),
        }
    )


__all__ = ["McapReader"]
