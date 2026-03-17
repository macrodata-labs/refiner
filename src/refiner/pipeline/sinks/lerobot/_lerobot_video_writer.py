from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from refiner.io import DataFolder
from refiner.media import Video
from refiner.pipeline.sinks.lerobot._lerobot_video_remux import (
    RemuxWriter,
    _VideoPtsAlignment,
    _VideoSourceProbe,
    probe_video_for_remux,
    probes_are_remux_compatible,
)
from refiner.pipeline.sinks.lerobot._lerobot_video_transcode import (
    TranscodeWriter,
    _resolve_video_fps,
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
class _VideoBatchItem:
    episode_index: int
    video: Video
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
class LeRobotVideoWriter:
    """Write one video column for one shard chunk."""

    folder: DataFolder
    chunk_key: str
    video_key: str
    video_config: "LeRobotVideoConfig"
    stats_config: "LeRobotStatsConfig"
    default_fps: int | None
    video_bytes_limit: int
    prepare_max_in_flight: int = 1
    preserve_order: bool = True

    _writer: RemuxWriter | TranscodeWriter | None = field(default=None, init=False)
    _next_file_index: int = field(default=0, init=False)

    async def write_videos(
        self,
        items: list[_VideoBatchItem],
        *,
        output_queue: asyncio.Queue[_CompletedVideoItem] | None = None,
    ) -> None:
        if not items:
            return None

        if not self.preserve_order:
            async for _, prepared_item in self._prepare_items(items):
                completed_item = await self._commit_item(prepared_item)
                if output_queue is not None:
                    await output_queue.put(completed_item)
                self._release_item_resources(prepared_item[0])
            return None

        prepared_by_index: dict[
            int, tuple[_VideoBatchItem, _VideoPtsAlignment | None]
        ] = {}
        next_commit_index = 0

        async for index, prepared in self._prepare_items(items):
            prepared_by_index[index] = prepared
            while next_commit_index in prepared_by_index:
                prepared_item = prepared_by_index.pop(next_commit_index)
                completed_item = await self._commit_item(prepared_item)
                if output_queue is not None:
                    await output_queue.put(completed_item)
                self._release_item_resources(prepared_item[0])
                next_commit_index += 1

        if prepared_by_index:
            raise RuntimeError("Prepared video items remained uncommitted")
        return None

    def _release_item_resources(self, item: _VideoBatchItem) -> None:
        item.source_stats = None

    def finalize(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None

    async def _prepare_items(
        self,
        items: list[_VideoBatchItem],
    ) -> AsyncIterator[tuple[int, tuple[_VideoBatchItem, _VideoPtsAlignment | None]]]:
        max_in_flight = max(1, int(self.prepare_max_in_flight))

        async def _prepare(
            index: int,
            item: _VideoBatchItem,
        ) -> tuple[int, tuple[_VideoBatchItem, _VideoPtsAlignment | None]]:
            return (
                index,
                (
                    item,
                    await probe_video_for_remux(
                        video=item.video,
                        source_stats=item.source_stats,
                        default_fps=self.default_fps,
                    ),
                ),
            )

        inflight: dict[
            asyncio.Task[tuple[int, tuple[_VideoBatchItem, _VideoPtsAlignment | None]]],
            int,
        ] = {}
        next_schedule_index = 0
        try:
            while next_schedule_index < len(items) or inflight:
                while (
                    next_schedule_index < len(items) and len(inflight) < max_in_flight
                ):
                    task = asyncio.create_task(
                        _prepare(next_schedule_index, items[next_schedule_index])
                    )
                    inflight[task] = next_schedule_index
                    next_schedule_index += 1

                done, _ = await asyncio.wait(
                    inflight.keys(),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    inflight.pop(task, None)
                    yield await task
        finally:
            for task in list(inflight):
                if not task.done():
                    task.cancel()
            if inflight:
                await asyncio.gather(*inflight.keys(), return_exceptions=True)

    async def _commit_item(
        self,
        prepared: tuple[_VideoBatchItem, _VideoPtsAlignment | None],
    ) -> _CompletedVideoItem:
        item, remux_alignment = prepared
        file_index = self._next_file_index
        if remux_alignment is not None:
            remux_writer = self._ensure_remux_writer(remux_alignment.probe)
            file_index = self._next_file_index
            from_timestamp, to_timestamp = remux_writer.append_video(
                item.video,
                remux_alignment,
            )
            stats = item.source_stats if item.source_stats is not None else {}
            feature = _video_feature(
                fps=remux_alignment.probe.fps,
                height=remux_alignment.probe.height,
                width=remux_alignment.probe.width,
                codec=remux_alignment.probe.codec or self.video_config.codec,
                pix_fmt=remux_alignment.probe.pix_fmt or self.video_config.pix_fmt,
            )
        else:
            transcode_writer = await self._ensure_transcode_writer(item.video)
            file_index = self._next_file_index
            (from_timestamp, to_timestamp), stats = await transcode_writer.append_video(
                video=item.video,
                stats_config=self.stats_config,
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

    async def _ensure_transcode_writer(self, video: Video) -> TranscodeWriter:
        fps = await _resolve_video_fps(
            video=video,
            video_key=self.video_key,
        )
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
    "_VideoBatchItem",
]
