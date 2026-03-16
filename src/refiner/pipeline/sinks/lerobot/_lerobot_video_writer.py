from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from refiner.execution.asyncio.runtime import submit
from refiner.execution.asyncio.window import AsyncWindow
from refiner.io import DataFolder
from refiner.media import Video
from refiner.pipeline.sinks.lerobot._lerobot_video import (
    VideoTrackWriter,
    _append_video_segment,
    _resolve_video_fps,
)
from refiner.pipeline.sinks.lerobot._lerobot_video_remux import (
    _PendingRemuxBatch,
    _PendingVideoRun,
    _PendingVideoSegment,
    _VideoProbeCache,
    probe_run_for_remux,
    probes_are_remux_compatible,
    remux_batch,
    run_can_extend,
)

if TYPE_CHECKING:
    from refiner.pipeline.sinks.lerobot._lerobot_writer import (
        LeRobotStatsConfig,
        LeRobotVideoConfig,
    )


_DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index}/file-{file_index:03d}.mp4"


def _format_video_path(*, chunk_key: str, video_key: str, file_idx: int) -> str:
    return _DEFAULT_VIDEO_PATH.format(
        video_key=video_key,
        chunk=chunk_key,
        chunk_key=chunk_key,
        chunk_index=chunk_key,
        file=file_idx,
        file_idx=file_idx,
        file_index=file_idx,
    )


@dataclass(slots=True)
class _CompletedVideoSegment:
    episode_index: int
    video_key: str
    video_meta: dict[str, Any]
    video_stats: dict[str, np.ndarray]


@dataclass(slots=True)
class _CompletedVideoRun:
    video_key: str
    feature: dict[str, Any] | None
    segments: list[_CompletedVideoSegment]


@dataclass(slots=True)
class VideoWriter:
    folder: DataFolder
    chunk_key: str
    video_key: str
    video_config: "LeRobotVideoConfig"
    stats_config: "LeRobotStatsConfig"
    default_fps: int | None
    video_bytes_limit: int

    _window: AsyncWindow[list[_CompletedVideoRun]] = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _probe_cache: _VideoProbeCache = field(default_factory=_VideoProbeCache, init=False)
    _pending_run: _PendingVideoRun | None = field(default=None, init=False)
    _pending_remux_batch: _PendingRemuxBatch | None = field(default=None, init=False)
    _track_writer: VideoTrackWriter | None = field(default=None, init=False)
    _next_file_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._window = AsyncWindow(max_in_flight=1, preserve_order=True)

    def submit(
        self,
        *,
        episode_index: int,
        video: Video,
        source_stats: dict[str, np.ndarray] | None,
    ) -> list[_CompletedVideoRun]:
        ready: list[list[_CompletedVideoRun]] = []
        current = self._pending_run
        if current is not None and not run_can_extend(current, video):
            ready = self._window.submit_blocking(self._export_closed_run(current))
            self._pending_run = None

        if self._pending_run is None:
            self._pending_run = _PendingVideoRun(video_key=self.video_key)

        self._pending_run.segments.append(
            _PendingVideoSegment(
                episode_index=episode_index,
                video=video,
                source_stats=source_stats,
            )
        )

        return [run for window in ready for run in window]

    def flush(self) -> list[_CompletedVideoRun]:
        ready: list[list[_CompletedVideoRun]] = []

        current = self._pending_run
        if current is not None and current.segments:
            ready.extend(self._window.submit_blocking(self._export_closed_run(current)))
            self._pending_run = None

        ready.extend(self._window.flush())

        completed: list[_CompletedVideoRun] = []
        for runs in ready:
            completed.extend(runs)

        batch = self._pending_remux_batch
        if batch is not None and batch.entries:
            completed.append(submit(self._flush_pending_remux_batch()).result())

        submit(self._close_writer()).result()
        return completed

    async def _export_closed_run(
        self, run: _PendingVideoRun
    ) -> list[_CompletedVideoRun]:
        probe = await asyncio.to_thread(
            probe_run_for_remux,
            run=run,
            probe_cache=self._probe_cache,
            default_fps=self.default_fps,
        )

        completed: list[_CompletedVideoRun] = []
        if probe is not None:
            batch = self._pending_remux_batch
            if (
                batch is not None
                and batch.entries
                and not probes_are_remux_compatible(batch.entries[-1][1], probe)
            ):
                completed.append(await self._flush_pending_remux_batch())
                batch = None
            if batch is None:
                batch = _PendingRemuxBatch()
                self._pending_remux_batch = batch
            batch.entries.append((run, probe))
            return completed

        batch = self._pending_remux_batch
        if batch is not None and batch.entries:
            completed.append(await self._flush_pending_remux_batch())

        completed.append(await self._transcode_run(run))
        return completed

    async def _flush_pending_remux_batch(self) -> _CompletedVideoRun:
        batch = self._pending_remux_batch
        if batch is None or not batch.entries:
            raise ValueError("remux batch requires runs and probes")
        self._pending_remux_batch = None

        async with self._lock:
            writer = self._track_writer
            if writer is not None:
                writer.close()
                self._track_writer = None
                self._next_file_index = writer.file_idx + 1

            file_idx = self._next_file_index
            self._next_file_index = file_idx + 1
            output_rel = _format_video_path(
                chunk_key=self.chunk_key,
                video_key=self.video_key,
                file_idx=file_idx,
            )

        await asyncio.to_thread(
            lambda: remux_batch(
                folder=self.folder,
                output_rel=output_rel,
                video_key=self.video_key,
                entries=batch.entries,
            )
        )

        duration_offset_s = 0.0
        segments: list[_CompletedVideoSegment] = []
        for run, _probe in batch.entries:
            run_start_s = float(run.segments[0].video.from_timestamp_s)
            run_duration_s = float(run.segments[-1].video.to_timestamp_s) - run_start_s
            for segment in run.segments:
                segments.append(
                    _CompletedVideoSegment(
                        episode_index=segment.episode_index,
                        video_key=self.video_key,
                        video_meta={
                            f"videos/{self.video_key}/chunk_index": self.chunk_key,
                            f"videos/{self.video_key}/file_index": file_idx,
                            f"videos/{self.video_key}/from_timestamp": (
                                duration_offset_s
                                + float(segment.video.from_timestamp_s)
                                - run_start_s
                            ),
                            f"videos/{self.video_key}/to_timestamp": (
                                duration_offset_s
                                + float(segment.video.to_timestamp_s)
                                - run_start_s
                            ),
                        },
                        video_stats=dict(segment.source_stats or {}),
                    )
                )
                self._probe_cache.cleanup_video(segment.video)
            duration_offset_s += run_duration_s

        probe = batch.entries[0][1]
        return _CompletedVideoRun(
            video_key=self.video_key,
            feature={
                "dtype": "video",
                "shape": [3, int(probe.height), int(probe.width)],
                "names": ["channels", "height", "width"],
                "info": {
                    "video.fps": int(probe.fps),
                    "video.height": int(probe.height),
                    "video.width": int(probe.width),
                    "video.channels": 3,
                    "video.codec": probe.codec or self.video_config.codec,
                    "video.pix_fmt": probe.pix_fmt or self.video_config.pix_fmt,
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            segments=segments,
        )

    async def _transcode_run(self, run: _PendingVideoRun) -> _CompletedVideoRun:
        completed_segments: list[_CompletedVideoSegment] = []
        feature: dict[str, Any] | None = None
        for segment in run.segments:
            video_meta, video_stats, feature = await self._write_video_segment(
                video=segment.video
            )
            completed_segments.append(
                _CompletedVideoSegment(
                    episode_index=segment.episode_index,
                    video_key=self.video_key,
                    video_meta=video_meta,
                    video_stats=video_stats,
                )
            )
            self._probe_cache.cleanup_video(segment.video)

        return _CompletedVideoRun(
            video_key=self.video_key,
            feature=feature,
            segments=completed_segments,
        )

    async def _write_video_segment(
        self,
        *,
        video: Video,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
        fps = await _resolve_video_fps(
            video=video,
            default_fps=self.default_fps,
            video_key=self.video_key,
        )

        async with self._lock:
            writer = self._track_writer
            if writer is not None and writer.file_idx != self._next_file_index:
                writer.close()
                self._track_writer = None
                writer = None

            if writer is None:
                output_rel = _format_video_path(
                    chunk_key=self.chunk_key,
                    video_key=self.video_key,
                    file_idx=self._next_file_index,
                )
                output_abs = self.folder._join(output_rel)
                self.folder.fs.makedirs(
                    self.folder.fs._parent(output_abs), exist_ok=True
                )
                output_file = self.folder.open(output_rel, mode="wb")
                writer = VideoTrackWriter(
                    chunk_key=self.chunk_key,
                    video_key=self.video_key,
                    file_idx=self._next_file_index,
                    fps=fps,
                    config=self.video_config,
                )
                try:
                    writer.open(output_file)
                except Exception:
                    output_file.close()
                    raise
                self._track_writer = writer

            if writer.size_bytes >= self.video_bytes_limit:
                writer.close()
                self._track_writer = None
                self._next_file_index = writer.file_idx + 1

                output_rel = _format_video_path(
                    chunk_key=self.chunk_key,
                    video_key=self.video_key,
                    file_idx=self._next_file_index,
                )
                output_abs = self.folder._join(output_rel)
                self.folder.fs.makedirs(
                    self.folder.fs._parent(output_abs), exist_ok=True
                )
                output_file = self.folder.open(output_rel, mode="wb")
                writer = VideoTrackWriter(
                    chunk_key=self.chunk_key,
                    video_key=self.video_key,
                    file_idx=self._next_file_index,
                    fps=fps,
                    config=self.video_config,
                )
                try:
                    writer.open(output_file)
                except Exception:
                    output_file.close()
                    raise
                self._track_writer = writer

            from_timestamp = float(writer.duration_s)
            clip_duration_s, clip_stats = await _append_video_segment(
                writer=writer,
                video=video,
                clip_from=video.from_timestamp_s,
                clip_to=video.to_timestamp_s,
                video_config=self.video_config,
                stats_config=self.stats_config,
            )
            to_timestamp = from_timestamp + float(clip_duration_s)
            writer.duration_s = to_timestamp
            stream = writer.stream
            if stream is None:
                raise RuntimeError("Video writer stream was not initialized")

            feature = {
                "dtype": "video",
                "shape": [3, int(stream.height), int(stream.width)],
                "names": ["channels", "height", "width"],
                "info": {
                    "video.fps": writer.fps,
                    "video.height": int(stream.height),
                    "video.width": int(stream.width),
                    "video.channels": 3,
                    "video.codec": self.video_config.codec,
                    "video.pix_fmt": self.video_config.pix_fmt,
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }

            if writer.output_file is not None:
                writer.output_file.flush()
            output_rel = _format_video_path(
                chunk_key=self.chunk_key,
                video_key=self.video_key,
                file_idx=writer.file_idx,
            )
            output_abs = self.folder._join(output_rel)
            size_bytes = writer.size_bytes
            try:
                size_bytes = max(
                    size_bytes,
                    int(self.folder.fs.info(output_abs).get("size", 0)),
                )
            except Exception:
                pass
            if size_bytes >= self.video_bytes_limit:
                writer.close()
                self._track_writer = None
                self._next_file_index = writer.file_idx + 1

            return (
                {
                    f"videos/{self.video_key}/chunk_index": self.chunk_key,
                    f"videos/{self.video_key}/file_index": writer.file_idx,
                    f"videos/{self.video_key}/from_timestamp": from_timestamp,
                    f"videos/{self.video_key}/to_timestamp": to_timestamp,
                },
                clip_stats,
                feature,
            )

    async def _close_writer(self) -> None:
        async with self._lock:
            writer = self._track_writer
            if writer is None:
                return
            writer.close()
            self._track_writer = None
            self._next_file_index = writer.file_idx + 1


__all__ = [
    "VideoWriter",
    "_CompletedVideoRun",
    "_CompletedVideoSegment",
]
