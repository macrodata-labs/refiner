from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable

import pyarrow as pa

from fsspec.spec import AbstractFileSystem

from refiner.execution.asyncio.window import AsyncWindow
from refiner.execution.operators.vectorized import iter_table_rows
from refiner.media import MediaFile, Video
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
    video_decoder_threads: int | None = None
    video_encoder_options: Mapping[str, str] | None = None
    enable_video_stats: bool = True
    video_stats_sample_stride: int = 1
    video_stats_quantile_bins: int = 500
    media_prelease_max_in_flight: int = 10
    media_prelease_preserve_order: bool = True

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
        if self.video_encoder_threads is not None and self.video_encoder_threads <= 0:
            raise ValueError("video_encoder_threads must be > 0 when provided")
        if self.video_decoder_threads is not None and self.video_decoder_threads <= 0:
            raise ValueError("video_decoder_threads must be > 0 when provided")
        if self.video_stats_sample_stride <= 0:
            raise ValueError("video_stats_sample_stride must be > 0")
        if self.video_stats_quantile_bins <= 1:
            raise ValueError("video_stats_quantile_bins must be > 1")
        if self.media_prelease_max_in_flight <= 0:
            raise ValueError("media_prelease_max_in_flight must be > 0")


class LeRobotWriterSink(BaseSink):
    """Stage 1 sink: write chunked data/video/stats artifacts."""

    def __init__(self, config: LeRobotWriterConfig):
        self.config = config
        self._writers: dict[str, _LeRobotShardWriter] = {}
        self._lease_window: AsyncWindow[tuple[Mapping[str, Any], str]] = AsyncWindow(
            max_in_flight=self.config.media_prelease_max_in_flight,
            preserve_order=self.config.media_prelease_preserve_order,
        )

    def cleanup_leases(self, row: Mapping[str, Any]) -> None:
        for key, value in row.items():
            if isinstance(value, Video) and isinstance(value.media, MediaFile):
                value.media.cleanup()

    def write_block(self, block: Block) -> ShardCounts:
        blocks_by_shard, counts = split_block_by_shard(block)

        for shard_id, shard_block in blocks_by_shard.items():
            if isinstance(shard_block, pa.Table):
                rows = iter_table_rows(shard_block)
            else:
                rows = shard_block

            for row in rows:
                self._lease_window.submit(self.pre_lease_row(row, shard_id))
                rows_leased = self._lease_window.drain(flush=False)
                self.process_leased_rows(rows_leased)

            # Non-video rows can otherwise stay deferred until shard completion,
            # which hides validation failures from direct sink use.
            rows_leased = self._lease_window.drain(flush=True)
            self.process_leased_rows(rows_leased)

        return counts

    async def pre_lease_row(
        self, row: Mapping[str, Any], shard_id: str
    ) -> tuple[Mapping[str, Any], str]:
        lease_requests = []
        for key, value in row.items():
            if isinstance(value, Video) and isinstance(value.media, MediaFile):
                lease_requests.append(
                    value.media.cache_file(cache_name=f"lerobot_writer:{key}")
                )
        await asyncio.gather(*lease_requests)
        return row, shard_id

    def process_leased_rows(
        self, rows: Iterable[tuple[Mapping[str, Any], str]]
    ) -> None:
        rank_raw = os.environ.get("REFINER_WORKER_RANK")
        rank = int(rank_raw) if rank_raw is not None else 0
        worker_id = "0" if rank < 0 else str(rank)
        for row, shard_id in rows:
            try:
                key = f"{worker_id}-{shard_id}"
                writer = self._writers.get(shard_id)
                if writer is None:
                    writer = _LeRobotShardWriter(config=self.config, chunk_key=key)
                    self._writers[shard_id] = writer
                writer.consume_row(row)
            finally:
                self.cleanup_leases(row)

    def on_shard_complete(self, shard_id: str) -> None:
        # Ensure we have no in-flight leases for this shard.
        rows_leased = self._lease_window.drain(flush=True)
        self.process_leased_rows(rows_leased)

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
