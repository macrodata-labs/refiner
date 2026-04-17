from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
import os
from typing import IO, Any
import numpy as np

from refiner.io import DataFolder
from refiner.utils import check_required_dependencies
from refiner.video.remux import (
    PreparedVideoSource,
    video_from_timestamp_s,
    video_to_timestamp_s,
)

_SEGMENTED_MP4_MOVFLAGS = "frag_keyframe+default_base_moof"
_FRAME_TIMESTAMP_EPSILON_S = 1e-6

FrameObserver = Callable[[int, np.ndarray], None]


@dataclass(frozen=True, slots=True)
class VideoTranscodeConfig:
    codec: str = "mpeg4"
    pix_fmt: str = "yuv420p"
    transencoding_threads: int | None = None
    videos_in_row: int = 1
    encoder_options: dict[str, str] | None = None

    @property
    def encoder_threads(self) -> int:
        return _resolve_threads(
            requested_threads=self.transencoding_threads,
            videos_in_row=self.videos_in_row,
        )

    @property
    def decoder_threads(self) -> int:
        return _resolve_threads(
            requested_threads=self.transencoding_threads,
            videos_in_row=self.videos_in_row,
        )

    def with_videos_in_row(self, videos_in_row: int) -> "VideoTranscodeConfig":
        return replace(self, videos_in_row=max(1, int(videos_in_row)))


class TranscodeWriter:
    def __init__(
        self,
        *,
        config: VideoTranscodeConfig,
        fps: int,
        output_file: IO[bytes],
    ) -> None:
        self.config = config
        self.fps = int(fps)
        self.output_file = output_file
        self.container: Any | None = None
        self.stream: Any | None = None
        self.frames_written = 0
        self.duration_s = 0.0

    @classmethod
    def open(
        cls,
        *,
        folder: DataFolder,
        output_rel: str,
        config: VideoTranscodeConfig,
        fps: int,
    ) -> "TranscodeWriter":
        check_required_dependencies("video transcoding", ["av"], dist="video")
        output_file = folder.open(output_rel, mode="wb")
        return cls.open_file(
            output_file=output_file,
            config=config,
            fps=fps,
        )

    @classmethod
    def open_file(
        cls,
        *,
        output_file: IO[bytes],
        config: VideoTranscodeConfig,
        fps: int,
        movflags: str | None = _SEGMENTED_MP4_MOVFLAGS,
    ) -> "TranscodeWriter":
        check_required_dependencies("video transcoding", ["av"], dist="video")
        import av

        writer = cls(config=config, fps=fps, output_file=output_file)
        try:
            options = {"movflags": movflags} if movflags is not None else None
            writer.container = av.open(
                output_file,
                mode="w",
                format="mp4",
                options=options,
            )
        except Exception:
            output_file.close()
            raise
        return writer

    @property
    def size_bytes(self) -> int:
        return 0 if self.output_file is None else int(self.output_file.tell())

    def ensure_stream(self, *, width: int, height: int) -> None:
        if self.stream is not None:
            return
        if self.container is None:
            raise RuntimeError("Video writer is not opened")

        options: dict[str, str] = dict(self.config.encoder_options or {})
        options.setdefault("threads", str(int(self.config.encoder_threads)))

        stream = self.container.add_stream(
            self.config.codec,
            rate=self.fps,
            options=options or None,
        )
        stream.width = int(width)
        stream.height = int(height)
        stream.pix_fmt = self.config.pix_fmt
        self.stream = stream

    def write_frame(self, frame: Any) -> None:
        if self.container is None or self.stream is None:
            raise RuntimeError("Video stream was not initialized")

        out_frame = frame
        if (
            frame.width != self.stream.width
            or frame.height != self.stream.height
            or frame.format.name != self.stream.pix_fmt
        ):
            out_frame = frame.reformat(
                width=self.stream.width,
                height=self.stream.height,
                format=self.stream.pix_fmt,
            )
        out_frame.pts = self.frames_written
        out_frame.time_base = Fraction(1, self.fps)

        for packet in self.stream.encode(out_frame):
            self.container.mux(packet)

        self.frames_written += 1
        self.duration_s = self.frames_written / float(self.fps)

    def append_prepared_video(
        self,
        *,
        prepared_source: PreparedVideoSource,
        frame_observer: FrameObserver | None = None,
    ) -> tuple[float, float]:
        from_timestamp = self.duration_s

        _configure_decoder(prepared_source.stream, self.config)
        for frame_index, frame in enumerate(
            _iter_selected_frames(
                container=prepared_source.container,
                stream=prepared_source.stream,
                clip_from=video_from_timestamp_s(prepared_source.video),
                clip_to=video_to_timestamp_s(prepared_source.video),
                seek=False,
            )
        ):
            self.ensure_stream(width=frame.width, height=frame.height)
            if frame_observer is not None:
                frame_observer(frame_index, frame.to_ndarray(format="rgb24"))
            self.write_frame(frame)

        if self.frames_written <= 0 or self.duration_s <= from_timestamp:
            raise ValueError("Video segment contains no decodable frames")

        return from_timestamp, self.duration_s

    def close(self) -> None:
        container = self.container
        if container is None:
            return
        try:
            if self.stream is not None:
                for packet in self.stream.encode(None):
                    container.mux(packet)
        finally:
            container.close()
            self.container = None
            self.stream = None
        if self.output_file is not None:
            self.output_file.close()
            self.output_file = None


def _configure_decoder(stream: Any, config: VideoTranscodeConfig) -> None:
    codec_context = getattr(stream, "codec_context", None)
    if codec_context is None:
        return
    if getattr(codec_context, "is_open", False):
        return
    codec_context.thread_count = int(config.decoder_threads)
    codec_context.thread_type = "AUTO"


def _cpu_thread_count() -> int:
    try:
        sched_getaffinity = getattr(os, "sched_getaffinity", None)
        if sched_getaffinity is None:
            raise AttributeError
        return max(1, len(sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _resolve_threads(
    *,
    requested_threads: int | None,
    videos_in_row: int,
) -> int:
    if requested_threads is None:
        requested_threads = _cpu_thread_count()
    return max(1, int(requested_threads) // max(1, int(videos_in_row)))


def _iter_selected_frames(
    *,
    container: Any,
    stream: Any,
    clip_from: float,
    clip_to: float | None,
    seek: bool,
):
    if seek and stream.time_base is not None:
        seek_ts = int(clip_from / float(stream.time_base))
        try:
            container.seek(seek_ts, stream=stream)
        except Exception:
            try:
                container.seek(max(0, seek_ts), any_frame=True, stream=stream)
            except Exception:
                pass

    for frame in container.decode(stream):
        ts = None
        if frame.pts is not None and frame.time_base is not None:
            ts = float(frame.pts * frame.time_base)
        if ts is None:
            continue
        if ts + _FRAME_TIMESTAMP_EPSILON_S < clip_from:
            continue
        if clip_to is not None and ts - _FRAME_TIMESTAMP_EPSILON_S >= clip_to:
            break
        yield frame


__all__ = [
    "FrameObserver",
    "TranscodeWriter",
    "VideoTranscodeConfig",
]
