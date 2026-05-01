from __future__ import annotations

from collections.abc import Iterator, Mapping
import posixpath
import tarfile
from typing import Any

from fsspec import AbstractFileSystem
import orjson

from refiner.io import DataFile
from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.datatype import DTypeMapping, dtype_to_plan
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.shard import FilePartsDescriptor
from refiner.pipeline.sources.readers.base import BaseReader, Shard, SourceUnit
from refiner.pipeline.sources.readers.utils import DEFAULT_TARGET_SHARD_BYTES


class WebDatasetReader(BaseReader):
    """WebDataset tar reader planned at archive granularity.

    Each output row is one WebDataset sample. Members are grouped by the path
    before the final extension, and the final extension becomes the output field
    name. JSON members are parsed to Python values by default; all other member
    payloads are emitted as bytes.
    """

    name = "read_webdataset"

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
        sample_key_column: str | None = "sample_key",
        parse_json: bool = True,
        dtypes: DTypeMapping | None = None,
    ):
        super().__init__(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            extensions=(".tar", ".tar.gz", ".tgz"),
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            file_path_column=file_path_column,
            split_by_bytes=False,
            dtypes=dtypes,
        )
        self.sample_key_column = sample_key_column
        self.parse_json = parse_json
        self._metadata_columns = frozenset(
            name for name in (file_path_column, sample_key_column) if name is not None
        )
        if (
            self.file_path_column is not None
            and self.sample_key_column is not None
            and self.file_path_column == self.sample_key_column
        ):
            raise ValueError("file_path_column and sample_key_column must be distinct")

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "sample_key_column": self.sample_key_column,
                "parse_json": self.parse_json,
                "dtypes": (
                    {key: dtype_to_plan(dtype) for key, dtype in self.dtypes.items()}
                    if self.dtypes
                    else None
                ),
            }
        )
        return description

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)
        for part in descriptor.parts:
            source = self.fileset.resolve_file(part.source_index, part.path)
            yield from self._read_archive(source)

    def _read_archive(self, source: DataFile) -> Iterator[SourceUnit]:
        current_key: str | None = None
        current_row: dict[str, Any] = {}

        def flush() -> Iterator[SourceUnit]:
            if current_key is None:
                return
            row = dict(current_row)
            if self.sample_key_column is not None:
                row[self.sample_key_column] = current_key
            yield DictRow(self._with_file_path(row, source))

        with (
            source.open(mode="rb") as raw,
            tarfile.open(fileobj=raw, mode="r|*") as tar,
        ):
            for member in tar:
                if not member.isfile():
                    continue
                member_path = posixpath.normpath(member.name).lstrip("/")
                if not member_path or member_path == ".":
                    continue
                directory, basename = posixpath.split(member_path)
                sample_prefix, separator, field_name = basename.partition(".")
                if not separator or not sample_prefix or not field_name:
                    continue
                sample_key = (
                    f"{directory}/{sample_prefix}" if directory else sample_prefix
                )
                field_name = field_name.lower()
                if current_key is not None and sample_key != current_key:
                    yield from flush()
                    current_row = {}
                current_key = sample_key
                if field_name in self._metadata_columns:
                    raise ValueError(
                        f"WebDataset member field {field_name!r} collides with a "
                        "metadata column; rename the metadata column or disable it"
                    )
                member_file = tar.extractfile(member)
                if member_file is None:
                    current_row[field_name] = b""
                    continue
                with member_file:
                    payload = member_file.read()
                if self.parse_json and (
                    field_name == "json" or field_name.endswith(".json")
                ):
                    try:
                        current_row[field_name] = orjson.loads(payload)
                    except orjson.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON member {member_path!r} in {source.abs_path()!r}"
                        ) from exc
                    continue
                current_row[field_name] = payload

        yield from flush()


__all__ = ["WebDatasetReader"]
