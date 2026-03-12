from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from fsspec.spec import AbstractFileSystem

from refiner.runtime.execution.vectorized import iter_table_rows
from refiner.runtime.sinks.base import BaseSink, Block, ShardCounts, split_block_by_shard
from ._lerobot_writer_shard import _LeRobotShardWriter


_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_DATA_FILE_SIZE_IN_MB = 100
_DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200


@dataclass(frozen=True, slots=True)
class LeRobotWriterConfig:
    root: str
    fs: AbstractFileSystem | None = None
    storage_options: Mapping[str, Any] | None = None
    overwrite: bool = False
    chunk_size: int = _DEFAULT_CHUNK_SIZE
    data_files_size_in_mb: int = _DEFAULT_DATA_FILE_SIZE_IN_MB
    video_files_size_in_mb: int = _DEFAULT_VIDEO_FILE_SIZE_IN_MB
    video_codec: str = "mpeg4"
    video_pix_fmt: str = "yuv420p"
    video_encoder_threads: int | None = None
    video_encoder_options: Mapping[str, str] | None = None
    enable_video_stats: bool = True
    video_stats_sample_stride: int = 1
    video_stats_quantile_bins: int = 500

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.data_files_size_in_mb <= 0:
            raise ValueError("data_files_size_in_mb must be > 0")
        if self.video_files_size_in_mb <= 0:
            raise ValueError("video_files_size_in_mb must be > 0")
        if not isinstance(self.video_codec, str) or not self.video_codec.strip():
            raise ValueError("video_codec must be a non-empty string")
        if not isinstance(self.video_pix_fmt, str) or not self.video_pix_fmt.strip():
            raise ValueError("video_pix_fmt must be a non-empty string")
        if self.video_encoder_threads is None:
            object.__setattr__(self, "video_encoder_threads", _cpu_thread_count())
        if self.video_encoder_threads is not None and self.video_encoder_threads <= 0:
            raise ValueError("video_encoder_threads must be > 0 when provided")
        if self.video_stats_sample_stride <= 0:
            raise ValueError("video_stats_sample_stride must be > 0")
        if self.video_stats_quantile_bins <= 1:
            raise ValueError("video_stats_quantile_bins must be > 1")


def _cpu_thread_count() -> int:
    try:
        sched_getaffinity = getattr(os, "sched_getaffinity", None)
        if sched_getaffinity is None:
            raise AttributeError
        return max(1, len(sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


class LeRobotWriterSink(BaseSink):
    """Stage 1 sink: write chunked data/video/stats artifacts."""

    def __init__(self, config: LeRobotWriterConfig):
        self.config = config
        self._writers: dict[str, _LeRobotShardWriter] = {}

    def write_block(self, block: Block) -> ShardCounts:
        blocks_by_shard, counts = split_block_by_shard(block)
        rank_raw = os.environ.get("REFINER_WORKER_RANK")
        rank = int(rank_raw) if rank_raw is not None else 0
        worker_id = "0" if rank < 0 else str(rank)

        for shard_id, shard_block in blocks_by_shard.items():
            if isinstance(shard_block, pa.Table):
                rows = iter_table_rows(shard_block)
            else:
                rows = shard_block

            key = f"{worker_id}-{shard_id}"
            writer = self._writers.get(shard_id)
            if writer is None:
                writer = _LeRobotShardWriter(config=self.config, chunk_key=key)
                self._writers[shard_id] = writer
            writer.consume_rows(rows)
        return counts

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
    "LeRobotWriterConfig",
    "LeRobotWriterSink",
]
