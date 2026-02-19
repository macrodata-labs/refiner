from collections.abc import Mapping, Sequence
from typing import Any, List, Literal

from fsspec import AbstractFileSystem

from refiner.io.fileset import DataFileSetLike
from refiner.processors.step import RefinerStep
from refiner.readers import CsvReader, JsonlReader, ParquetReader
from refiner.readers.base import BaseReader
from refiner.readers.utils import DEFAULT_TARGET_SHARD_BYTES


class RefinerPipeline:
    source: BaseReader
    pipeline_steps: List[RefinerStep]

    def __init__(
        self, source: BaseReader, pipeline_steps: List[RefinerStep] | None = None
    ):
        self.source: BaseReader = source
        self.pipeline_steps = list(pipeline_steps) if pipeline_steps else []

    def __add_step(self, step: RefinerStep) -> "RefinerPipeline":
        return self.__class__(self.source, self.pipeline_steps + [step])


## readers
def read_csv(
    inputs: DataFileSetLike,
    *,
    fs: AbstractFileSystem | None = None,
    storage_options: Mapping[str, Any] | None = None,
    recursive: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    multiline_rows: bool = False,
    encoding: str = "utf-8",
    sharding_mode: Literal["bytes_lazy", "scan"] = "bytes_lazy",
) -> RefinerPipeline:
    return RefinerPipeline(
        source=CsvReader(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            target_shard_bytes=target_shard_bytes,
            multiline_rows=multiline_rows,
            encoding=encoding,
            sharding_mode=sharding_mode,
        )
    )


def read_jsonl(
    inputs: DataFileSetLike,
    *,
    fs: AbstractFileSystem | None = None,
    storage_options: Mapping[str, Any] | None = None,
    recursive: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
) -> RefinerPipeline:
    return RefinerPipeline(
        source=JsonlReader(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            target_shard_bytes=target_shard_bytes,
        )
    )


def read_parquet(
    inputs: DataFileSetLike,
    *,
    fs: AbstractFileSystem | None = None,
    storage_options: Mapping[str, Any] | None = None,
    recursive: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    arrow_batch_size: int = 65536,
    columns_to_read: Sequence[str] | None = None,
    sharding_mode: Literal["rowgroups", "bytes_lazy"] = "rowgroups",
) -> RefinerPipeline:
    return RefinerPipeline(
        source=ParquetReader(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            target_shard_bytes=target_shard_bytes,
            arrow_batch_size=arrow_batch_size,
            columns_to_read=columns_to_read,
            sharding_mode=sharding_mode,
        )
    )
