from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from fsspec.spec import AbstractFileSystem

from refiner.execution.operators.vectorized import iter_table_rows
from refiner.pipeline.sinks.base import (
    BaseSink,
    Block,
    ShardCounts,
    split_block_by_shard,
)
from refiner.pipeline.sinks.lerobot._lerobot_writer_shard import _LeRobotShardWriter


_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_DATA_FILE_SIZE_IN_MB = 100
_DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200


@dataclass(frozen=True, slots=True)
class LeRobotVideoConfig:
    codec: str = "mpeg4"
    pix_fmt: str = "yuv420p"
    encoder_threads: int | None = None
    decoder_threads: int | None = None
    encoder_options: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.codec, str) or not self.codec.strip():
            raise ValueError("video.codec must be a non-empty string")
        if not isinstance(self.pix_fmt, str) or not self.pix_fmt.strip():
            raise ValueError("video.pix_fmt must be a non-empty string")
        if self.encoder_threads is not None and self.encoder_threads <= 0:
            raise ValueError("video.encoder_threads must be > 0 when provided")
        if self.decoder_threads is not None and self.decoder_threads <= 0:
            raise ValueError("video.decoder_threads must be > 0 when provided")


@dataclass(frozen=True, slots=True)
class LeRobotStatsConfig:
    sample_stride: int = 1
    quantile_bins: int = 500

    def __post_init__(self) -> None:
        if self.sample_stride <= 0:
            raise ValueError("stats.sample_stride must be > 0")
        if self.quantile_bins <= 1:
            raise ValueError("stats.quantile_bins must be > 1")


@dataclass(frozen=True, slots=True)
class LeRobotWriterConfig:
    root: str
    fs: AbstractFileSystem | None = None
    storage_options: Mapping[str, Any] | None = None
    overwrite: bool = False
    chunk_size: int = _DEFAULT_CHUNK_SIZE
    data_files_size_in_mb: int = _DEFAULT_DATA_FILE_SIZE_IN_MB
    video_files_size_in_mb: int = _DEFAULT_VIDEO_FILE_SIZE_IN_MB
    video: LeRobotVideoConfig = field(default_factory=LeRobotVideoConfig)
    stats: LeRobotStatsConfig = field(default_factory=LeRobotStatsConfig)
    media_prelease_max_in_flight: int = 10
    media_prelease_preserve_order: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.data_files_size_in_mb <= 0:
            raise ValueError("data_files_size_in_mb must be > 0")
        if self.video_files_size_in_mb <= 0:
            raise ValueError("video_files_size_in_mb must be > 0")
        if self.media_prelease_max_in_flight <= 0:
            raise ValueError("media_prelease_max_in_flight must be > 0")


class LeRobotWriterSink(BaseSink):
    """Stage 1 sink: write chunked data/video/stats artifacts."""

    def __init__(self, config: LeRobotWriterConfig):
        self.config = config
        self._writers: dict[str, _LeRobotShardWriter] = {}

    def write_block(self, block: Block) -> ShardCounts:
        blocks_by_shard, counts = split_block_by_shard(block)

        for shard_id, shard_block in blocks_by_shard.items():
            if isinstance(shard_block, pa.Table):
                rows = iter_table_rows(shard_block)
            else:
                rows = shard_block

            for row in rows:
                self.process_row(row, shard_id)

        return counts

    def process_row(self, row: Mapping[str, Any], shard_id: str) -> None:
        rank_raw = os.environ.get("REFINER_WORKER_RANK")
        rank = int(rank_raw) if rank_raw is not None else 0
        worker_id = "0" if rank < 0 else str(rank)
        key = f"{worker_id}-{shard_id}"
        writer = self._writers.get(shard_id)
        if writer is None:
            writer = _LeRobotShardWriter(config=self.config, chunk_key=key)
            self._writers[shard_id] = writer
        writer.consume_row(row)

    def on_shard_complete(self, shard_id: str) -> None:
        writer = self._writers.pop(shard_id, None)
        if writer is None:
            return
        writer.finalize()

    def close(self) -> None:
        for writer in self._writers.values():
            writer.finalize()
        self._writers.clear()


__all__ = [
    "LeRobotStatsConfig",
    "LeRobotVideoConfig",
    "LeRobotWriterConfig",
    "LeRobotWriterSink",
]
