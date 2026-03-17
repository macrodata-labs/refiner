from __future__ import annotations

from fractions import Fraction
from typing import IO, TYPE_CHECKING, Any, Iterator

import av
import numpy as np

from refiner.io import DataFolder
from refiner.media import VideoFile
from refiner.pipeline.sinks.lerobot._lerobot_stats import _RunningQuantileStats
from refiner.pipeline.sinks.lerobot._lerobot_video_remux import _PreparedSource
from refiner.pipeline.sinks.lerobot._lerobot_video_types import (
    video_from_timestamp_s,
    video_to_timestamp_s,
)

if TYPE_CHECKING:
    from refiner.pipeline.sinks.lerobot._lerobot_writer import (
        LeRobotStatsConfig,
        LeRobotVideoConfig,
    )


_SEGMENTED_MP4_MOVFLAGS = "frag_keyframe+default_base_moof"
_FRAME_TIMESTAMP_EPSILON_S = 1e-6


class TranscodeWriter:
    def __init__(
        self,
        *,
        config: "LeRobotVideoConfig",
        video_key: str,
        fps: int,
        output_file: IO[bytes],
    ) -> None:
        self.config = config
        self.video_key = video_key
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
        config: "LeRobotVideoConfig",
        video_key: str,
        fps: int,
    ) -> TranscodeWriter:
        output_file = folder.open(output_rel, mode="wb")
        writer = cls(
            config=config,
            video_key=video_key,
            fps=fps,
            output_file=output_file,
        )
        writer.container = av.open(
            output_file,
            mode="w",
            format="mp4",
            options={"movflags": _SEGMENTED_MP4_MOVFLAGS},
        )
        return writer

    @property
    def size_bytes(self) -> int:
        return 0 if self.output_file is None else int(self.output_file.tell())

    def ensure_stream(self, *, width: int, height: int) -> None:
        if self.stream is not None:
            return
        if self.container is None:
            raise RuntimeError("Video writer is not opened")

        options: dict[str, str] = {}
        if self.config.encoder_options:
            for key, value in self.config.encoder_options.items():
                options[str(key)] = str(value)
        if self.config.encoder_threads is not None:
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

    def write_frame(self, frame: av.VideoFrame) -> None:
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
        video: VideoFile,
        prepared_source: _PreparedSource,
        stats_config: "LeRobotStatsConfig",
    ) -> tuple[tuple[float, float], dict[str, np.ndarray]]:
        from_timestamp = self.duration_s
        tracker = _new_tracker(stats_config)
        selected_frames = 0

        _configure_decoder(prepared_source.stream, self.config)
        clip_from = video_from_timestamp_s(video)
        clip_to = video_to_timestamp_s(video)
        for frame_index, frame in enumerate(
            _iter_selected_frames(
                container=prepared_source.container,
                stream=prepared_source.stream,
                clip_from=clip_from,
                clip_to=clip_to,
                seek=False,
            )
        ):
            self.ensure_stream(width=frame.width, height=frame.height)
            self.write_frame(frame)
            _update_video_stats_if_due(
                tracker=tracker,
                frame_index=frame_index,
                sample_stride=stats_config.sample_stride,
                rgb_frame=frame.to_ndarray(format="rgb24"),
            )
            selected_frames += 1

        if selected_frames <= 0:
            raise ValueError(
                f"Video segment for {video.uri!r} contains no decodable frames in "
                f"[{clip_from:.6f}, {clip_to if clip_to is not None else 'end'})."
            )
        stats = _video_stats_from_tracker(tracker)
        return (from_timestamp, self.duration_s), stats

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


def _configure_decoder(stream: Any, video_config: "LeRobotVideoConfig") -> None:
    codec_context = getattr(stream, "codec_context", None)
    if codec_context is None or video_config.decoder_threads is None:
        return
    codec_context.thread_count = int(video_config.decoder_threads)
    codec_context.thread_type = "AUTO"


def _iter_selected_frames(
    *,
    container: Any,
    stream: Any,
    clip_from: float,
    clip_to: float | None,
    seek: bool,
) -> Iterator[av.VideoFrame]:
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


def _new_tracker(stats_config: "LeRobotStatsConfig") -> _RunningQuantileStats:
    return _RunningQuantileStats(
        [0.01, 0.10, 0.50, 0.90, 0.99],
        num_quantile_bins=stats_config.quantile_bins,
    )


def _update_video_stats_tracker(
    *,
    tracker: _RunningQuantileStats,
    image_chw: np.ndarray,
) -> None:
    if image_chw.ndim != 3:
        return
    pixels = np.transpose(image_chw, (1, 2, 0)).reshape(-1, image_chw.shape[0])
    tracker.update(pixels)


def _auto_downsample_height_width(
    image_chw: np.ndarray,
    *,
    target_size: int = 150,
    max_size_threshold: int = 300,
) -> np.ndarray:
    _, height, width = image_chw.shape
    if max(width, height) < max_size_threshold:
        return image_chw
    factor = max(1, int((width if width > height else height) / target_size))
    return image_chw[:, ::factor, ::factor]


def _update_video_stats_if_due(
    *,
    tracker: _RunningQuantileStats,
    frame_index: int,
    sample_stride: int,
    rgb_frame: np.ndarray | None,
) -> None:
    if frame_index % sample_stride != 0 or rgb_frame is None or rgb_frame.ndim != 3:
        return
    _update_video_stats_tracker(
        tracker=tracker,
        image_chw=_auto_downsample_height_width(np.transpose(rgb_frame, (2, 0, 1))),
    )


def _video_stats_from_tracker(
    tracker: _RunningQuantileStats,
) -> dict[str, np.ndarray]:
    if tracker.count <= 0:
        return {}
    stats = tracker.get_statistics()
    normalized: dict[str, np.ndarray] = {}
    for key, value in stats.items():
        if key == "count":
            normalized[key] = value
        else:
            normalized[key] = (
                np.asarray(value, dtype=np.float64).reshape(3, 1, 1) / 255.0
            )
    return normalized
