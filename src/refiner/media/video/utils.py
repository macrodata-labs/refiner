from __future__ import annotations

from fractions import Fraction
import os
import tempfile
import subprocess
from typing import Literal

import av
import numpy as np
from .decoder_cache import get_video_decoder_cache


def slice_video_to_mp4_bytes(
    *,
    local_path: str,
    from_timestamp_s: float,
    to_timestamp_s: float | None,
    extract_backend: Literal["pyav", "ffmpeg"] = "pyav",
) -> bytes:
    start_ts = max(0.0, float(from_timestamp_s))
    end_ts = float(to_timestamp_s) if to_timestamp_s is not None else None
    if end_ts is not None and end_ts <= start_ts:
        raise ValueError(
            "Invalid video timestamp bounds: "
            f"from_timestamp_s={start_ts}, to_timestamp_s={end_ts}"
        )

    if extract_backend == "ffmpeg":
        return _slice_video_to_mp4_bytes_ffmpeg(
            local_path=local_path,
            from_timestamp_s=start_ts,
            to_timestamp_s=end_ts,
        )
    return _slice_video_to_mp4_bytes_pyav(
        local_path=local_path,
        from_timestamp_s=start_ts,
        to_timestamp_s=end_ts,
    )


def _slice_video_to_mp4_bytes_pyav(
    *,
    local_path: str,
    from_timestamp_s: float,
    to_timestamp_s: float | None,
) -> bytes:
    out_path = None

    fd, out_path = tempfile.mkstemp(prefix="refiner_video_slice_", suffix=".mp4")
    os.close(fd)

    selected_frames = 0

    try:
        with av.open(local_path) as input_container:
            input_stream = input_container.streams.video[0]
            fps = resolve_video_fps(input_stream=input_stream)

            with av.open(out_path, mode="w", options={"movflags": "faststart"}) as out:
                output_stream: av.video.stream.VideoStream | None = None
                for frame in input_container.decode(input_stream):
                    ts = None
                    if frame.pts is not None and frame.time_base is not None:
                        ts = float(frame.pts * frame.time_base)
                    if ts is None:
                        continue
                    if ts + 1e-6 < start_ts:
                        continue
                    if end_ts is not None and ts - 1e-6 >= end_ts:
                        break

                    if output_stream is None:
                        output_stream = out.add_stream("mpeg4", rate=fps)
                        output_stream.width = frame.width
                        output_stream.height = frame.height
                        output_stream.pix_fmt = "yuv420p"

                    out_frame = frame.reformat(
                        width=output_stream.width,
                        height=output_stream.height,
                        format=output_stream.pix_fmt,
                    )
                    out_frame.pts = selected_frames
                    out_frame.time_base = Fraction(1, fps)
                    for packet in output_stream.encode(out_frame):
                        out.mux(packet)
                    selected_frames += 1

                if output_stream is not None:
                    for packet in output_stream.encode(None):
                        out.mux(packet)

        if selected_frames <= 0:
            raise ValueError(
                "Video segment contains no decodable frames in requested timestamp window."
            )
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        _cleanup_temp_path(out_path)


def _slice_video_to_mp4_bytes_ffmpeg(
    *,
    local_path: str,
    from_timestamp_s: float,
    to_timestamp_s: float | None,
) -> bytes:
    start_ts = max(0.0, float(from_timestamp_s))
    end_ts = float(to_timestamp_s) if to_timestamp_s is not None else None

    out_path = None
    fd, out_path = tempfile.mkstemp(prefix="refiner_video_slice_ffmpeg_", suffix=".mp4")
    os.close(fd)

    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start_ts),
        "-i",
        local_path,
    ]
    if end_ts is not None:
        cmd += ["-to", str(end_ts)]
    cmd += [
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_non_negative",
        "-movflags",
        "+faststart",
        out_path,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg binary not found. Install ffmpeg or use extract_backend='pyav'."
        ) from exc
    except subprocess.CalledProcessError as exc:
        _cleanup_temp_path(out_path)
        raise ValueError(
            "ffmpeg failed while extracting video segment: "
            f"{exc.stderr.decode(errors='replace')}"
        ) from exc

    try:
        if os.path.getsize(out_path) <= 0:
            raise ValueError(
                "ffmpeg produced an empty video segment for requested window."
            )
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        _cleanup_temp_path(out_path)


def decode_video_segment_frames(
    *,
    local_path: str,
    from_timestamp_s: float,
    to_timestamp_s: float | None,
    decoder_cache_name: str = "default",
    decode_backend: Literal["pyav", "ffmpeg"] = "pyav",
) -> tuple[tuple[np.ndarray, ...], int | None, int | None, str | None]:
    if decode_backend == "ffmpeg":
        return _decode_video_segment_frames_with_ffmpeg_segment(
            local_path=local_path,
            from_timestamp_s=from_timestamp_s,
            to_timestamp_s=to_timestamp_s,
        )

    decoder_cache = get_video_decoder_cache(name=decoder_cache_name)
    return decoder_cache.decode_segment(
        local_path=local_path,
        from_timestamp_s=from_timestamp_s,
        to_timestamp_s=to_timestamp_s,
    )


def _decode_video_segment_frames_with_ffmpeg_segment(
    *,
    local_path: str,
    from_timestamp_s: float,
    to_timestamp_s: float | None,
) -> tuple[tuple[np.ndarray, ...], int | None, int | None, str | None] | None:
    segment_bytes = _slice_video_to_mp4_bytes_ffmpeg(
        local_path=local_path,
        from_timestamp_s=from_timestamp_s,
        to_timestamp_s=to_timestamp_s,
    )

    fd, segment_path = tempfile.mkstemp(prefix="refiner_video_segment_", suffix=".mp4")
    os.close(fd)
    try:
        with open(segment_path, "wb") as f:
            f.write(segment_bytes)

        frames: list[np.ndarray] = []
        width: int | None = None
        height: int | None = None
        with av.open(segment_path) as input_container:
            input_stream = input_container.streams.video[0]
            for frame in input_container.decode(input_stream):
                frame_rgb = frame.to_ndarray(format="rgb24")
                if width is None:
                    width = frame.width
                    height = frame.height
                frames.append(frame_rgb)

        if not frames:
            raise ValueError(
                "Video segment contains no decodable frames in requested timestamp window."
            )

        return tuple(frames), width, height, "rgb24"
    finally:
        _cleanup_temp_path(segment_path)


def resolve_video_fps(*, input_stream: av.video.stream.VideoStream) -> int:
    avg_rate = input_stream.average_rate
    if avg_rate is not None:
        avg_fps = float(avg_rate)
        if avg_fps > 0:
            return max(1, int(round(avg_fps)))
    guessed = input_stream.guessed_rate
    if guessed is not None:
        guessed_fps = float(guessed)
        if guessed_fps > 0:
            return max(1, int(round(guessed_fps)))
    base = input_stream.base_rate
    if base is not None:
        base_fps = float(base)
        if base_fps > 0:
            return max(1, int(round(base_fps)))
    return 30


def _cleanup_temp_path(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


__all__ = [
    "slice_video_to_mp4_bytes",
    "decode_video_segment_frames",
    "resolve_video_fps",
]
