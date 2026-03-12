from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping

import av
import numpy as np

from refiner.media import DecodedVideo, Video

from ._lerobot_stats import _RunningQuantileStats


def _to_rel_path(folder: Any, abs_path: str) -> str:
    root = folder.path.rstrip("/")
    prefix = f"{root}/"
    if abs_path.startswith(prefix):
        return abs_path[len(prefix) :]
    return abs_path


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


def _resolve_video_fps(video: Video, default_fps: int | None) -> int:
    if video.fps is not None and int(video.fps) > 0:
        return int(video.fps)
    if default_fps is not None and int(default_fps) > 0:
        return int(default_fps)

    if isinstance(video.media, DecodedVideo):
        return 30

    with video.media.cached_path(suffix=".mp4") as local_path:
        with av.open(local_path) as container:
            stream = container.streams.video[0]
            if stream.average_rate is not None:
                try:
                    rate = float(stream.average_rate)
                except (TypeError, ValueError):
                    pass
                else:
                    fps = int(round(rate))
                    if fps > 0:
                        return fps
            if stream.base_rate is not None:
                try:
                    rate = float(stream.base_rate)
                except (TypeError, ValueError):
                    pass
                else:
                    fps = int(round(rate))
                    if fps > 0:
                        return fps
    return 30


@dataclass(slots=True)
class _VideoWriterState:
    chunk_idx: str
    file_idx: int
    temp_path: str
    container: Any
    stream: Any | None
    fps: int
    frames_written: int = 0
    duration_s: float = 0.0
    size_bytes: int = 0


def _create_video_writer(
    *,
    video_key: str,
    chunk_idx: str,
    file_idx: int,
    fps: int,
) -> _VideoWriterState:
    fd, temp_path = tempfile.mkstemp(
        prefix=f"refiner_lerobot_{video_key.replace('/', '_')}_",
        suffix=".mp4",
    )
    os.close(fd)
    container = av.open(temp_path, mode="w", options={"movflags": "faststart"})
    return _VideoWriterState(
        chunk_idx=chunk_idx,
        file_idx=file_idx,
        temp_path=temp_path,
        container=container,
        stream=None,
        fps=fps,
    )


def _ensure_video_output_stream(
    *,
    state: _VideoWriterState,
    config: Mapping[str, Any],
    width: int,
    height: int,
) -> None:
    if state.stream is not None:
        return

    stream_options: dict[str, str] = {}
    video_encoder_options = config.get("video_encoder_options")
    if video_encoder_options:
        for key, value in video_encoder_options.items():
            stream_options[str(key)] = str(value)

    threads = config.get("video_encoder_threads")
    if threads is not None:
        stream_options.setdefault("threads", str(int(threads)))

    stream = state.container.add_stream(
        str(config["video_codec"]),
        rate=state.fps,
        options=stream_options or None,
    )
    stream.width = int(width)
    stream.height = int(height)
    stream.pix_fmt = str(config["video_pix_fmt"])
    state.stream = stream


def _flush_video_writer(
    *,
    state: _VideoWriterState,
    folder: Any,
    video_path_template: str,
    video_key: str,
) -> None:
    try:
        if state.stream is not None:
            for packet in state.stream.encode(None):
                state.container.mux(packet)
        state.container.close()

        rel = _format_chunk_path(
            video_path_template,
            chunk=state.chunk_idx,
            file_idx=state.file_idx,
            video_key=video_key,
        )
        with (
            open(state.temp_path, "rb") as src,
            folder.open(rel, mode="wb") as dst,
        ):
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    finally:
        try:
            os.remove(state.temp_path)
        except FileNotFoundError:
            pass


def _append_video_segment(
    *,
    state: _VideoWriterState,
    video: Video,
    clip_from: float,
    clip_to: float,
    video_config: Mapping[str, Any],
) -> tuple[float, dict[str, np.ndarray] | None]:
    tracker = _RunningQuantileStats([0.01, 0.10, 0.50, 0.90, 0.99])

    if isinstance(video.media, DecodedVideo):
        duration_s = _append_video_segment_from_frames(
            state=state,
            video=video.media,
            video_config=video_config,
            tracker=tracker,
        )
        return duration_s, _video_stats_from_tracker(tracker)

    selected_frames = 0
    epsilon = 1e-6
    with video.media.cached_path(suffix=".mp4") as local_path:
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

                _ensure_video_output_stream(
                    state=state,
                    config=video_config,
                    width=frame.width,
                    height=frame.height,
                )

                if state.stream is None:
                    raise RuntimeError("Video stream was not initialized")

                out_frame = frame.reformat(
                    width=state.stream.width,
                    height=state.stream.height,
                    format=state.stream.pix_fmt,
                )
                out_frame.pts = state.frames_written
                out_frame.time_base = Fraction(1, state.fps)
                for packet in state.stream.encode(out_frame):
                    state.container.mux(packet)

                state.frames_written += 1
                selected_frames += 1
                rgb = frame.to_ndarray(format="rgb24")
                chw = np.transpose(rgb, (2, 0, 1))
                _update_video_stats_tracker(
                    tracker=tracker,
                    image_chw=_auto_downsample_height_width(chw),
                )

    if selected_frames <= 0:
        raise ValueError(
            f"Video segment for {video.uri!r} contains no decodable frames in "
            f"[{clip_from:.6f}, {clip_to if clip_to is not None else 'end'})."
        )

    return (selected_frames / float(state.fps), _video_stats_from_tracker(tracker))


def _append_video_segment_from_frames(
    *,
    state: _VideoWriterState,
    video: DecodedVideo,
    video_config: Mapping[str, Any],
    tracker: _RunningQuantileStats,
) -> float:
    if video.frame_count <= 0:
        raise ValueError(
            f"Decoded video segment for {video.uri!r} contains no decodable frames."
        )

    frames_written = 0
    width = video.width
    height = video.height
    for frame_data in video.frames:
        if width is None or height is None:
            if isinstance(frame_data, np.ndarray) and frame_data.ndim >= 2:
                height = int(frame_data.shape[0])
                width = int(frame_data.shape[1])
        if width is None or height is None:
            raise RuntimeError("Decoded video frame shape missing width/height metadata.")

        _ensure_video_output_stream(
            state=state,
            config=video_config,
            width=width,
            height=height,
        )

        if state.stream is None:
            raise RuntimeError("Video stream not initialized")

        out_frame = av.VideoFrame.from_ndarray(
            frame_data,
            format=video.pix_fmt or "rgb24",
        )
        out_frame.pts = state.frames_written
        out_frame.time_base = Fraction(1, state.fps)
        out_frame = out_frame.reformat(
            width=state.stream.width,
            height=state.stream.height,
            format=state.stream.pix_fmt,
        )
        for packet in state.stream.encode(out_frame):
            state.container.mux(packet)

        state.frames_written += 1
        frames_written += 1
        if isinstance(frame_data, np.ndarray) and frame_data.ndim == 3:
            chw = np.transpose(frame_data, (2, 0, 1))
            _update_video_stats_tracker(
                tracker=tracker,
                image_chw=_auto_downsample_height_width(chw),
            )

    return float(frames_written) / float(state.fps)


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


def _video_stats_from_tracker(
    tracker: _RunningQuantileStats,
) -> dict[str, np.ndarray] | None:
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
