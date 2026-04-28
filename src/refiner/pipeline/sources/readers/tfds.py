from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from refiner.pipeline.data.shard import RowRangeDescriptor, Shard
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.base import BaseSource, SourceUnit
from refiner.pipeline.sources.readers._tensorflow import (
    require_tfds,
    tensorflow_batch_to_table,
)

_DEFAULT_EXAMPLES_PER_SHARD = 10_000


class TfdsReader(BaseSource):
    """TensorFlow Datasets reader that preserves TFDS feature decoding."""

    name = "read_tfds"

    def __init__(
        self,
        name: str,
        *,
        config: str | None = None,
        split: str = "train",
        data_dir: str | None = None,
        download: bool = False,
        batch_size: int = 1024,
        examples_per_shard: int = _DEFAULT_EXAMPLES_PER_SHARD,
        num_shards: int | None = None,
        shuffle_files: bool = False,
        read_config: Any | None = None,
        decoders: Mapping[str, Any] | None = None,
        as_supervised: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if examples_per_shard <= 0:
            raise ValueError("examples_per_shard must be > 0")
        if num_shards is not None and num_shards <= 0:
            raise ValueError("num_shards must be > 0 when provided")
        _, self.tfds = require_tfds()
        self.dataset_name = name
        self.config = config
        self.split = split
        self.batch_size = int(batch_size)
        self.examples_per_shard = int(examples_per_shard)
        self.num_shards = num_shards
        self.shuffle_files = shuffle_files
        self.read_config = read_config
        self.decoders = dict(decoders) if decoders else None
        self.as_supervised = as_supervised
        self.builder = self.tfds.builder(name, config=config, data_dir=data_dir)
        if download:
            self.builder.download_and_prepare()
        if split not in self.builder.info.splits:
            raise ValueError(
                "read_tfds currently shards plain split names only; pass a split "
                f"from builder.info.splits, got {split!r}"
            )
        self.num_examples = int(self.builder.info.splits[split].num_examples)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset_name,
            "config": self.config,
            "split": self.split,
            "batch_size": self.batch_size,
            "examples_per_shard": self.examples_per_shard,
            "num_shards": self.num_shards,
        }

    def list_shards(self) -> list[Shard]:
        shards: list[Shard] = []
        if self.num_shards is None:
            for start in range(0, self.num_examples, self.examples_per_shard):
                shards.append(
                    Shard.from_row_range(
                        start=start,
                        end=min(start + self.examples_per_shard, self.num_examples),
                        global_ordinal=len(shards),
                    )
                )
            return shards
        for ordinal in range(self.num_shards):
            start = ordinal * self.num_examples // self.num_shards
            end = (ordinal + 1) * self.num_examples // self.num_shards
            if start != end:
                shards.append(
                    Shard.from_row_range(
                        start=start,
                        end=end,
                        global_ordinal=len(shards),
                    )
                )
        return shards

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        if not isinstance(descriptor, RowRangeDescriptor):
            raise TypeError("TfdsReader requires row-range shards")
        split = f"{self.split}[{descriptor.start}:{descriptor.end}]"
        dataset = self.builder.as_dataset(
            split=split,
            batch_size=self.batch_size,
            shuffle_files=self.shuffle_files,
            read_config=self.read_config,
            decoders=self.decoders,
            as_supervised=self.as_supervised,
        )
        for batch in dataset.prefetch(1):
            if self.as_supervised:
                batch = {"input": batch[0], "target": batch[1]}
            table = tensorflow_batch_to_table(batch)
            if table.num_rows > 0:
                yield Tabular(table)


__all__ = ["TfdsReader"]
