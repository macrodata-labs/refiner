from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from refiner.execution.asyncio.window import AsyncWindow
from refiner.io.datafolder import DataFolderLike
from refiner.pipeline.sinks.base import (
    BaseSink,
    Block,
    ShardCounts,
    describe_datafolder_path,
    split_block_by_shard,
)
from refiner.pipeline.sinks.lerobot._lerobot_writer_shard import _LeRobotShardWriter
from refiner.worker.context import get_active_run_handle

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
    quantile_bins: int = 5000
    force_recompute_video_stats: bool = False

    def __post_init__(self) -> None:
        if self.sample_stride <= 0:
            raise ValueError("stats.sample_stride must be > 0")
        if self.quantile_bins <= 1:
            raise ValueError("stats.quantile_bins must be > 1")


@dataclass(frozen=True, slots=True)
class LeRobotWriterConfig:
    output: DataFolderLike
    data_files_size_in_mb: int = _DEFAULT_DATA_FILE_SIZE_IN_MB
    video_files_size_in_mb: int = _DEFAULT_VIDEO_FILE_SIZE_IN_MB
    video: LeRobotVideoConfig = field(default_factory=LeRobotVideoConfig)
    stats: LeRobotStatsConfig = field(default_factory=LeRobotStatsConfig)
    max_video_prepare_in_flight: int = 10

    def __post_init__(self) -> None:
        if self.data_files_size_in_mb <= 0:
            raise ValueError("data_files_size_in_mb must be > 0")
        if self.video_files_size_in_mb <= 0:
            raise ValueError("video_files_size_in_mb must be > 0")
        if self.max_video_prepare_in_flight <= 0:
            raise ValueError("max_video_prepare_in_flight must be > 0")


class LeRobotWriterSink(BaseSink):
    """Stage 1 sink: write chunked data/video/stats artifacts."""

    def __init__(self, config: LeRobotWriterConfig):
        self.config = config
        self._writers: dict[str, _LeRobotShardWriter] = {}
        self._async_window = AsyncWindow[None](
            max_in_flight=self.config.max_video_prepare_in_flight,
            preserve_order=False,
        )

    def write_block(self, block: Block) -> ShardCounts:
        blocks_by_shard, counts = split_block_by_shard(block)

        for shard_id, shard_block in blocks_by_shard.items():
            for row in shard_block:
                self._async_window.submit_blocking(
                    self._writer_for_shard(shard_id).write_row(row=row)
                )

        return counts

    def _writer_for_shard(self, shard_id: str) -> _LeRobotShardWriter:
        writer = self._writers.get(shard_id)
        token = get_active_run_handle().worker_token
        key = f"{shard_id}__w{token}"
        if writer is None:
            writer = _LeRobotShardWriter(config=self.config, chunk_key=key)
            self._writers[shard_id] = writer
        return writer

    def on_shard_complete(self, shard_id: str) -> None:
        writer = self._writers.pop(shard_id, None)
        # TODO: We don't have to flush the whole thing we just need to flush the shard rows
        self._async_window.flush()
        if writer is None:
            return
        writer.finalize()

    def close(self) -> None:
        self._async_window.flush()
        for writer in self._writers.values():
            writer.finalize()
        self._writers.clear()

    def describe(self) -> tuple[str, str, dict[str, object]]:
        return (
            "write_lerobot",
            "writer",
            {
                "path": describe_datafolder_path(self.config.output),
                "data_files_size_in_mb": self.config.data_files_size_in_mb,
                "video_files_size_in_mb": self.config.video_files_size_in_mb,
                "max_video_prepare_in_flight": self.config.max_video_prepare_in_flight,
            },
        )


__all__ = [
    "LeRobotStatsConfig",
    "LeRobotVideoConfig",
    "LeRobotWriterConfig",
    "LeRobotWriterSink",
]
