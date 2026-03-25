from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Literal
from dataclasses import dataclass, field
from functools import partial

from refiner.execution.asyncio.runtime import io_executor
from refiner.io import DataFolder
from refiner.video.remux import (
    PreparedVideoSource,
    RemuxWriter,
    VideoSourceProbe,
    prepare_video_source,
    probes_are_remux_compatible,
)
from refiner.video.transcode import (
    FrameObserver,
    TranscodeWriter,
    VideoTranscodeConfig,
)
from refiner.video.types import VideoFile


@dataclass(frozen=True, slots=True)
class WrittenVideoSegment:
    stream_key: str
    file_index: int
    output_rel: str
    from_timestamp: float
    to_timestamp: float
    fps: int
    width: int
    height: int
    codec: str
    pix_fmt: str


@dataclass(frozen=True, slots=True)
class WrittenVideo:
    segment: WrittenVideoSegment
    mode: Literal["remux", "transcode"]


@dataclass(slots=True)
class VideoStreamWriter:
    folder: DataFolder
    stream_key: str
    transcode_config: VideoTranscodeConfig
    video_bytes_limit: int
    output_rel_template: str
    output_context: Mapping[str, str | int] = field(default_factory=dict)

    _writer: RemuxWriter | TranscodeWriter | None = field(default=None, init=False)
    _next_file_index: int = field(default=0, init=False)
    _commit_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def write_video(
        self,
        video: VideoFile,
        *,
        frame_observer: FrameObserver | None = None,
        force_transcode: bool = False,
    ) -> WrittenVideo:
        prepared = await prepare_video_source(cache_key=self.stream_key, video=video)
        try:
            return await self._commit(
                prepared,
                frame_observer=frame_observer,
                force_transcode=force_transcode,
            )
        finally:
            prepared.close()

    def close(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None

    async def close_async(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._commit_lock:
            await loop.run_in_executor(io_executor(), self.close)

    async def _commit(
        self,
        prepared: PreparedVideoSource,
        *,
        frame_observer: FrameObserver | None,
        force_transcode: bool,
    ) -> WrittenVideo:
        async with self._commit_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                io_executor(),
                partial(
                    self._commit_sync,
                    prepared,
                    frame_observer=frame_observer,
                    force_transcode=force_transcode,
                ),
            )

    def _commit_sync(
        self,
        prepared: PreparedVideoSource,
        *,
        frame_observer: FrameObserver | None,
        force_transcode: bool,
    ) -> WrittenVideo:
        if not self._should_transcode(
            prepared,
            frame_observer=frame_observer,
            force_transcode=force_transcode,
        ):
            probe = prepared.probe
            if probe is None:
                raise RuntimeError("Remux path selected without a source probe")
            if probe.fps is None:
                raise ValueError("Prepared remux item is missing FPS")
            writer = self._ensure_remux_writer(probe)
            file_index = self._next_file_index
            from_timestamp, to_timestamp = writer.append_prepared_video(prepared)
            segment = WrittenVideoSegment(
                stream_key=self.stream_key,
                file_index=file_index,
                output_rel=self._current_output_rel,
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
                fps=int(probe.fps),
                width=int(probe.width),
                height=int(probe.height),
                codec=str(probe.codec or self.transcode_config.codec),
                pix_fmt=str(probe.pix_fmt or self.transcode_config.pix_fmt),
            )
            if writer.size_bytes >= self.video_bytes_limit:
                self._rotate_writer()
            return WrittenVideo(segment=segment, mode="remux")

        fps = self._transcode_fps(prepared)
        if fps is None:
            raise ValueError("Prepared transcode item is missing FPS")
        writer = self._ensure_transcode_writer(fps)
        file_index = self._next_file_index
        from_timestamp, to_timestamp = writer.append_prepared_video(
            prepared_source=prepared,
            frame_observer=frame_observer,
        )
        if writer.stream is None:
            raise RuntimeError("Transcode writer did not initialize an output stream")
        segment = WrittenVideoSegment(
            stream_key=self.stream_key,
            file_index=file_index,
            output_rel=self._current_output_rel,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            fps=int(writer.fps),
            width=int(writer.stream.width),
            height=int(writer.stream.height),
            codec=self.transcode_config.codec,
            pix_fmt=self.transcode_config.pix_fmt,
        )
        if writer.size_bytes >= self.video_bytes_limit:
            self._rotate_writer()
        return WrittenVideo(segment=segment, mode="transcode")

    def _should_transcode(
        self,
        prepared: PreparedVideoSource,
        *,
        frame_observer: FrameObserver | None,
        force_transcode: bool,
    ) -> bool:
        _ = frame_observer
        return force_transcode or prepared.probe is None or prepared.alignment is None

    def _transcode_fps(self, prepared: PreparedVideoSource) -> int | None:
        if prepared.probe is not None and prepared.probe.fps is not None:
            return int(prepared.probe.fps)
        if isinstance(self._writer, TranscodeWriter):
            return int(self._writer.fps)
        return None

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
            config=self.transcode_config,
            fps=fps,
        )
        self._writer = writer
        return writer

    def _ensure_remux_writer(self, probe: VideoSourceProbe) -> RemuxWriter:
        writer = self._writer
        if isinstance(writer, RemuxWriter) and probes_are_remux_compatible(
            writer.probe,
            probe,
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

    @property
    def _current_output_rel(self) -> str:
        return self.output_rel_template.format(
            file_index=self._next_file_index,
            stream_key=self.stream_key,
            **self.output_context,
        )

    def _rotate_writer(self) -> None:
        writer = self._writer
        if writer is not None:
            writer.close()
            self._writer = None
        self._next_file_index += 1


__all__ = [
    "VideoStreamWriter",
    "WrittenVideo",
    "WrittenVideoSegment",
]
