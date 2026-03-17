from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from refiner.execution.asyncio.runtime import io_executor
from refiner.io import DataFolder
from refiner.media import VideoFile
from refiner.pipeline.sinks.lerobot._lerobot_video_remux import (
    RemuxWriter,
    _PreparedSource,
    _VideoSourceProbe,
    prepare_video,
    probes_are_remux_compatible,
)
from refiner.pipeline.sinks.lerobot._lerobot_video_transcode import (
    TranscodeWriter,
)

if TYPE_CHECKING:
    from refiner.pipeline.sinks.lerobot._lerobot_writer import (
        LeRobotStatsConfig,
        LeRobotVideoConfig,
    )


DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index}/file-{file_index:03d}.mp4"


def _video_feature(
    *,
    fps: int,
    height: int,
    width: int,
    codec: str,
    pix_fmt: str,
) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [3, int(height), int(width)],
        "names": ["channels", "height", "width"],
        "info": {
            "video.fps": int(fps),
            "video.height": int(height),
            "video.width": int(width),
            "video.channels": 3,
            "video.codec": codec,
            "video.pix_fmt": pix_fmt,
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


@dataclass(slots=True)
class _VideoItem:
    episode_index: int
    video: VideoFile
    source_stats: dict[str, np.ndarray] | None = None


@dataclass(slots=True)
class _CompletedVideoSegment:
    episode_index: int
    video_key: str
    file_index: int
    chunk_key: str
    from_timestamp: float
    to_timestamp: float
    stats: dict[str, np.ndarray]


@dataclass(slots=True)
class _CompletedVideoItem:
    video_key: str
    feature: dict[str, Any] | None
    segment: _CompletedVideoSegment


@dataclass(slots=True)
class _PreparedVideoItem:
    item: _VideoItem
    source: _PreparedSource
    transcode_fps: int | None = None


@dataclass(slots=True)
class LeRobotVideoWriter:
    """Write one video column for one shard chunk."""

    folder: DataFolder
    chunk_key: str
    video_key: str
    video_config: "LeRobotVideoConfig"
    stats_config: "LeRobotStatsConfig"
    default_fps: int | None
    video_bytes_limit: int

    _writer: RemuxWriter | TranscodeWriter | None = field(default=None, init=False)
    _next_file_index: int = field(default=0, init=False)
    _commit_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def write_video(
        self,
        video: VideoFile,
        *,
        episode_index: int,
        source_stats: dict[str, np.ndarray] | None = None,
    ) -> _CompletedVideoItem:
        prepared = await self._prepare_item(
            _VideoItem(
                episode_index=episode_index,
                video=video,
                source_stats=source_stats,
            )
        )
        try:
            return await self._commit_item(prepared)
        finally:
            self._release_prepared_resources(prepared)

    def _release_prepared_resources(self, prepared: _PreparedVideoItem) -> None:
        prepared.item.source_stats = None
        prepared.source.close()

    def finalize(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None

    async def _prepare_item(
        self,
        item: _VideoItem,
    ) -> _PreparedVideoItem:
        prepared_source = await prepare_video(
            video=item.video,
            default_fps=self.default_fps,
        )
        return _PreparedVideoItem(
            item=item,
            source=prepared_source,
            transcode_fps=(
                None
                if prepared_source.alignment is not None
                or prepared_source.probe is None
                else prepared_source.probe.fps
            ),
        )

    async def _commit_item(
        self,
        prepared: _PreparedVideoItem,
    ) -> _CompletedVideoItem:
        # We need the lock so that we can rotate the writer without race conditions
        async with self._commit_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                io_executor(),
                partial(self._commit_item_sync, prepared),
            )

    def _commit_item_sync(
        self,
        prepared: _PreparedVideoItem,
    ) -> _CompletedVideoItem:
        item = prepared.item
        source = prepared.source
        if source.alignment is not None and source.probe is not None:
            remux_writer = self._ensure_remux_writer(source.probe)
            file_index = self._next_file_index
            from_timestamp, to_timestamp = remux_writer.append_prepared_video(
                source,
            )
            stats = item.source_stats if item.source_stats is not None else {}
            feature = _video_feature(
                fps=source.probe.fps,
                height=source.probe.height,
                width=source.probe.width,
                codec=source.probe.codec or self.video_config.codec,
                pix_fmt=source.probe.pix_fmt or self.video_config.pix_fmt,
            )
        else:
            transcode_fps = prepared.transcode_fps
            if transcode_fps is None:
                raise ValueError("Prepared transcode item is missing FPS")
            transcode_writer = self._ensure_transcode_writer(transcode_fps)
            file_index = self._next_file_index
            (from_timestamp, to_timestamp), stats = (
                transcode_writer.append_prepared_video(
                    video=item.video,
                    prepared_source=source,
                    stats_config=self.stats_config,
                )
            )
            if transcode_writer.stream is None:
                raise RuntimeError(
                    "Transcode writer did not initialize an output stream"
                )
            feature = _video_feature(
                fps=int(transcode_writer.fps),
                height=int(transcode_writer.stream.height),
                width=int(transcode_writer.stream.width),
                codec=self.video_config.codec,
                pix_fmt=self.video_config.pix_fmt,
            )

        return _CompletedVideoItem(
            video_key=self.video_key,
            feature=feature,
            segment=_CompletedVideoSegment(
                episode_index=item.episode_index,
                video_key=self.video_key,
                file_index=file_index,
                chunk_key=self.chunk_key,
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
                stats=stats,
            ),
        )

    def _ensure_transcode_writer(self, fps: int) -> TranscodeWriter:
        writer = self._writer
        if isinstance(writer, TranscodeWriter) and writer.fps == fps:
            if writer.size_bytes >= self.video_bytes_limit:
                self._rotate_writer()
            else:
                return writer

        if self._writer is not None:
            self._rotate_writer()

        writer = TranscodeWriter.open(
            folder=self.folder,
            output_rel=self._current_output_rel,
            config=self.video_config,
            video_key=self.video_key,
            fps=fps,
        )
        self._writer = writer
        return writer

    def _ensure_remux_writer(
        self,
        probe: _VideoSourceProbe,
    ) -> RemuxWriter:
        writer = self._writer
        if isinstance(writer, RemuxWriter) and probes_are_remux_compatible(
            writer.probe, probe
        ):
            if writer.size_bytes >= self.video_bytes_limit:
                self._rotate_writer()
            else:
                return writer

        if self._writer is not None:
            self._rotate_writer()

        writer = RemuxWriter.open(
            folder=self.folder,
            output_rel=self._current_output_rel,
            probe=probe,
        )
        self._writer = writer
        return writer

    def _rotate_writer(self) -> None:
        writer = self._writer
        if writer is None:
            return
        writer.close()
        self._writer = None
        self._next_file_index += 1

    @property
    def _current_output_rel(self) -> str:
        return DEFAULT_VIDEO_PATH.format(
            video_key=self.video_key,
            chunk=self.chunk_key,
            chunk_key=self.chunk_key,
            chunk_index=self.chunk_key,
            file=self._next_file_index,
            file_idx=self._next_file_index,
            file_index=self._next_file_index,
        )


__all__ = [
    "DEFAULT_VIDEO_PATH",
    "LeRobotVideoWriter",
    "_CompletedVideoItem",
    "_CompletedVideoSegment",
    "_VideoItem",
]
