from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import IO, Any, Mapping

import av
import numpy as np

from refiner.io.datafile import DataFile
from refiner.media import DecodedVideo, Video
from refiner.pipeline.sinks.lerobot._lerobot_stats import _RunningQuantileStats
from refiner.pipeline.utils.cache.decoder_cache import get_video_decoder_cache
from refiner.pipeline.utils.cache.file_cache import get_media_cache


_SEGMENTED_MP4_MOVFLAGS = "frag_keyframe+empty_moov+default_base_moof"


class _TrackedOutputFile:
    def __init__(self, wrapped: IO[bytes], owner: "VideoTrackWriter") -> None:
        self._wrapped = wrapped
        self._owner = owner

    def write(self, data: bytes) -> int:
        written = self._wrapped.write(data)
        if written is None:
            written = len(data)
        self._owner.size_bytes += int(written)
        return int(written)

    def flush(self) -> None:
        if hasattr(self._wrapped, "flush"):
            self._wrapped.flush()

    def close(self) -> None:
        self._wrapped.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


@dataclass(slots=True)
class VideoTrackWriter:
    chunk_key: str
    video_key: str
    file_idx: int
    fps: int
    config: Mapping[str, Any]

    container: Any | None = None
    stream: Any | None = None
    output_file: Any | None = None
    frames_written: int = 0
    duration_s: float = 0.0
    size_bytes: int = 0

    def __post_init__(self) -> None:
        self.fps = int(self.fps)

    def open(self, output: IO[bytes]) -> None:
        if self.container is not None:
            return

        tracked_output = _TrackedOutputFile(output, self)
        self.output_file = tracked_output
        self.container = av.open(
            tracked_output,
            mode="w",
            format="mp4",
            options={"movflags": _SEGMENTED_MP4_MOVFLAGS},
        )

    def ensure_stream(self, *, width: int, height: int) -> None:
        if self.stream is not None:
            return
        if self.container is None:
            raise RuntimeError("Video writer is not opened")

        stream_options: dict[str, str] = {}
        video_encoder_options = self.config.get("video_encoder_options")
        if video_encoder_options:
            for key, value in video_encoder_options.items():
                stream_options[str(key)] = str(value)

        threads = self.config.get("video_encoder_threads")
        if threads is not None:
            stream_options.setdefault("threads", str(int(threads)))

        stream = self.container.add_stream(
            str(self.config["video_codec"]),
            rate=self.fps,
            options=stream_options or None,
        )
        stream.width = int(width)
        stream.height = int(height)
        stream.pix_fmt = str(self.config["video_pix_fmt"])
        self.stream = stream

    def write_frame(self, frame: av.VideoFrame) -> None:
        if self.container is None or self.stream is None:
            raise RuntimeError("Video stream was not initialized")

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

        output_file = self.output_file
        if output_file is not None:
            output_file.close()
            self.output_file = None


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


def _update_video_stats_tracker(
    *,
    tracker: _RunningQuantileStats,
    image_chw: np.ndarray,
) -> None:
    if image_chw.ndim != 3:
        return

    pixels = np.transpose(image_chw, (1, 2, 0)).reshape(-1, image_chw.shape[0])
    tracker.update(pixels)


def _update_video_stats_if_due(
    *,
    tracker: _RunningQuantileStats | None,
    frame_index: int,
    sample_stride: int,
    rgb_frame: np.ndarray | None,
) -> None:
    if tracker is None or frame_index % sample_stride != 0:
        return
    if rgb_frame is None:
        return
    if rgb_frame.ndim != 3:
        return
    _update_video_stats_tracker(
        tracker=tracker,
        image_chw=_auto_downsample_height_width(np.transpose(rgb_frame, (2, 0, 1))),
    )


def _video_stats_from_tracker(
    tracker: _RunningQuantileStats | None,
) -> dict[str, np.ndarray] | None:
    if tracker is None:
        return None
    if tracker.count <= 0:
        return None

    ft_stats = tracker.get_statistics()
    normalized: dict[str, np.ndarray] = {}
    for key, value in ft_stats.items():
        if key == "count":
            normalized[key] = value
            continue
        normalized[key] = np.asarray(value, dtype=np.float64).reshape(3, 1, 1) / 255.0
    return normalized


async def _resolve_video_fps(
    *,
    video: Video,
    default_fps: int | None,
    video_key: str,
) -> int:
    if video.fps is not None:
        return int(round(float(video.fps)))
    if default_fps is not None:
        return int(round(float(default_fps)))

    if isinstance(video.media, DecodedVideo):
        return 30

    data_file = DataFile.resolve(video.media.uri)
    cache_name = f"lerobot_writer:{video_key}"
    media_cache = get_media_cache(name=cache_name)
    decoder_cache = get_video_decoder_cache(name=cache_name, media_cache=media_cache)
    resolved_fps = await decoder_cache.resolve_fps(data_file=data_file)
    if resolved_fps is not None:
        return resolved_fps
    return 30


async def _append_video_segment(
    *,
    writer: VideoTrackWriter,
    video: Video,
    video_key: str,
    clip_from: float,
    clip_to: float,
    video_config: Mapping[str, Any],
) -> tuple[float, dict[str, np.ndarray] | None]:
    enable_video_stats = bool(video_config.get("enable_video_stats", True))
    sample_stride = int(video_config.get("video_stats_sample_stride", 1))
    tracker: _RunningQuantileStats | None = None
    if enable_video_stats:
        tracker = _RunningQuantileStats(
            [0.01, 0.10, 0.50, 0.90, 0.99],
            num_quantile_bins=int(video_config.get("video_stats_quantile_bins", 500)),
        )

    if isinstance(video.media, DecodedVideo):
        duration_s = _append_video_segment_from_frames(
            writer=writer,
            video=video.media,
            tracker=tracker,
            sample_stride=sample_stride,
        )
        return duration_s, _video_stats_from_tracker(tracker)

    duration_s = await _append_video_segment_from_media(
        writer=writer,
        video=video,
        video_key=video_key,
        clip_from=clip_from,
        clip_to=clip_to,
        tracker=tracker,
        sample_stride=sample_stride,
    )
    return duration_s, _video_stats_from_tracker(tracker)


async def _append_video_segment_from_media(
    *,
    writer: VideoTrackWriter,
    video: Video,
    clip_from: float,
    clip_to: float,
    tracker: _RunningQuantileStats | None,
    sample_stride: int,
    video_key: str,
) -> float:
    selected_frames = 0
    frame_index = 0
    data_file = DataFile.resolve(video.media.uri)
    cache_name = f"lerobot_writer:{video_key}"
    media_cache = get_media_cache(name=cache_name)
    decoder_cache = get_video_decoder_cache(name=cache_name, media_cache=media_cache)

    def _on_frame(frame: av.VideoFrame) -> None:
        nonlocal selected_frames, frame_index
        rgb_frame = frame.to_ndarray(format="rgb24")
        writer.ensure_stream(width=frame.width, height=frame.height)
        writer.write_frame(av.VideoFrame.from_ndarray(rgb_frame, format="rgb24"))
        _update_video_stats_if_due(
            tracker=tracker,
            frame_index=frame_index,
            sample_stride=sample_stride,
            rgb_frame=rgb_frame,
        )
        selected_frames += 1
        frame_index += 1

    try:
        await decoder_cache.decode_segment_with_callback_from_data_file(
            data_file=data_file,
            from_timestamp_s=clip_from,
            to_timestamp_s=clip_to,
            on_frame=_on_frame,
        )
    except ValueError as exc:
        if "contains no decodable frames" not in str(exc):
            raise
        raise ValueError(
            f"Video segment for {video.uri!r} contains no decodable frames in "
            f"[{clip_from:.6f}, {clip_to if clip_to is not None else 'end'})."
        ) from exc

    if selected_frames <= 0:
        raise ValueError(
            f"Video segment for {video.uri!r} contains no decodable frames in "
            f"[{clip_from:.6f}, {clip_to if clip_to is not None else 'end'})."
        )

    return float(selected_frames) / float(writer.fps)


def _append_video_segment_from_frames(
    *,
    writer: VideoTrackWriter,
    video: DecodedVideo,
    tracker: _RunningQuantileStats | None,
    sample_stride: int,
) -> float:
    if video.frame_count <= 0:
        raise ValueError(
            f"Decoded video segment for {video.uri!r} contains no decodable frames."
        )

    width = video.width
    height = video.height
    frames_written = 0
    for frame_data in video.frames:
        if width is None or height is None:
            if isinstance(frame_data, np.ndarray) and frame_data.ndim >= 2:
                height = int(frame_data.shape[0])
                width = int(frame_data.shape[1])
        if width is None or height is None:
            raise RuntimeError(
                "Decoded video frame shape missing width/height metadata."
            )

        writer.ensure_stream(width=width, height=height)
        frame = av.VideoFrame.from_ndarray(
            frame_data,
            format=video.pix_fmt or "rgb24",
        )
        writer.write_frame(frame)

        frames_written += 1
        _update_video_stats_if_due(
            tracker=tracker,
            frame_index=frames_written - 1,
            sample_stride=sample_stride,
            rgb_frame=frame_data if isinstance(frame_data, np.ndarray) else None,
        )

    return float(frames_written) / float(writer.fps)
