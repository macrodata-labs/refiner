from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import IO, Any, Mapping

import av
import numpy as np

from refiner.io import DataFolder
from refiner.io.datafile import DataFile
from refiner.media import DecodedVideo, Video
from refiner.media.cache import get_media_cache

from ._lerobot_stats import _RunningQuantileStats


_SEGMENTED_MP4_MOVFLAGS = "frag_keyframe+empty_moov+default_base_moof"
_VIDEO_STATS_SAMPLE_STRIDE = 1
_VIDEO_STATS_QUANTILE_BINS = 500


def _format_chunk_path(
    template: str,
    *,
    chunk: str,
    file_idx: int,
    video_key: str | None = None,
) -> str:
    return template.format(
        video_key="" if video_key is None else video_key,
        chunk=chunk,
        chunk_key=chunk,
        chunk_index=chunk,
        file=file_idx,
        file_idx=file_idx,
        file_index=file_idx,
    )


def _to_rel_path(folder: DataFolder, abs_path: str) -> str:
    root = folder.path.rstrip("/")
    prefix = f"{root}/"
    if abs_path.startswith(prefix):
        return abs_path[len(prefix) :]
    return abs_path


def _coerce_positive_fps(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return None
    if fps <= 0:
        return None
    return int(round(fps))


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
    output_file: IO[bytes] | None = None
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


def _resolve_video_fps(video: Video, default_fps: int | None) -> int:
    resolved_video_fps = _coerce_positive_fps(video.fps)
    if resolved_video_fps is not None:
        return resolved_video_fps
    default_resolved_fps = _coerce_positive_fps(default_fps)
    if default_resolved_fps is not None:
        return default_resolved_fps

    if isinstance(video.media, DecodedVideo):
        return 30

    data_file = DataFile.resolve(video.media.uri)
    with get_media_cache("lerobot_writer").cached(file=data_file) as local_path:
        with av.open(local_path) as container:
            stream = container.streams.video[0]
            for rate in (stream.average_rate, stream.base_rate):
                fps = _coerce_positive_fps(rate)
                if fps is not None:
                    return fps
    return 30


async def _append_video_segment(
    *,
    writer: VideoTrackWriter,
    video: Video,
    clip_from: float,
    clip_to: float,
    video_config: Mapping[str, Any],
) -> tuple[float, dict[str, np.ndarray] | None]:
    sample_stride = _VIDEO_STATS_SAMPLE_STRIDE
    tracker = _RunningQuantileStats(
        [0.01, 0.10, 0.50, 0.90, 0.99],
        num_quantile_bins=_VIDEO_STATS_QUANTILE_BINS,
    )

    if isinstance(video.media, DecodedVideo):
        duration_s = _append_video_segment_from_frames(
            writer=writer,
            video=video.media,
            video_config=video_config,
            tracker=tracker,
            sample_stride=sample_stride,
        )
        return duration_s, _video_stats_from_tracker(tracker)

    duration_s = await _append_video_segment_from_media(
        writer=writer,
        video=video,
        clip_from=clip_from,
        clip_to=clip_to,
        video_config=video_config,
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
    video_config: Mapping[str, Any],
    tracker: _RunningQuantileStats | None,
    sample_stride: int,
) -> float:
    selected_frames = 0
    frame_index = 0
    epsilon = 1e-6
    # TODO: Fix the cache here
    cache_name = f"lerobot_writer:{video.video_key}"
    data_file = DataFile.resolve(video.media.uri)
    async with get_media_cache(cache_name).cached(file=data_file) as local_path:
        with av.open(local_path) as input_container:
            input_stream = input_container.streams.video[0]
            for frame in input_container.decode(input_stream):
                if frame.pts is None or frame.time_base is None:
                    continue

                ts = float(frame.pts * frame.time_base)
                if ts + epsilon < clip_from:
                    continue
                if ts - epsilon >= clip_to:
                    break

                writer.ensure_stream(width=frame.width, height=frame.height)
                rgb_frame = None
                if tracker is not None:
                    rgb_frame = frame.to_ndarray(format="rgb24")
                    writer.write_frame(
                        av.VideoFrame.from_ndarray(
                            rgb_frame,
                            format="rgb24",
                        )
                    )
                else:
                    writer.write_frame(frame)

                selected_frames += 1
                _update_video_stats_if_due(
                    tracker=tracker,
                    frame_index=frame_index,
                    sample_stride=sample_stride,
                    rgb_frame=rgb_frame,
                )
                frame_index += 1

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
    video_config: Mapping[str, Any],
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
            raise RuntimeError("Decoded video frame shape missing width/height metadata.")

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
