from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import av
import numpy as np

from refiner.execution.asyncio.runtime import submit
from refiner.io import DataFolder
from refiner.media import MediaFile, Video
from refiner.media.video.types import DecodedVideo

_TIMESTAMP_EPSILON_S = 1e-3
_VIDEO_PROBE_CACHE_MAX_ENTRIES = 256
_SEGMENTED_MP4_MOVFLAGS = "frag_keyframe+default_base_moof"


@dataclass(slots=True)
class _PendingVideoSegment:
    episode_index: int
    video: Video
    source_stats: dict[str, np.ndarray] | None = None


@dataclass(slots=True)
class _PendingVideoRun:
    video_key: str
    segments: list[_PendingVideoSegment] = field(default_factory=list)


@dataclass(slots=True)
class _PendingRemuxBatch:
    entries: list[tuple[_PendingVideoRun, _VideoSourceProbe]] = field(
        default_factory=list
    )


@dataclass(frozen=True, slots=True)
class _VideoSourceProbe:
    local_path: str
    duration_s: float
    width: int
    height: int
    fps: int
    time_base: Fraction
    codec: str | None
    pix_fmt: str | None
    has_audio: bool
    packet_boundary_pts: tuple[int, ...]
    keyframe_pts: tuple[int, ...]
    monotonic_pts_dts: bool


def _stream_duration_seconds(container: Any, stream: Any) -> float | None:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration / av.time_base)
    return None


def _match_timestamp_to_pts(
    timestamp_s: float,
    candidates: Sequence[int],
    time_base: Fraction,
) -> int | None:
    for candidate in candidates:
        if (
            abs((float(candidate) * float(time_base)) - float(timestamp_s))
            <= _TIMESTAMP_EPSILON_S
        ):
            return int(candidate)
    return None


def run_can_extend(run: _PendingVideoRun, video: Video) -> bool:
    if not run.segments:
        return True

    prev = run.segments[-1].video
    if isinstance(prev.media, DecodedVideo) or isinstance(video.media, DecodedVideo):
        return False
    return prev.uri == video.uri and (
        abs(float(prev.to_timestamp_s) - float(video.from_timestamp_s))
        <= _TIMESTAMP_EPSILON_S
    )


def _run_is_full_source(
    run: _PendingVideoRun,
    probe: _VideoSourceProbe,
) -> bool:
    if not run.segments:
        return False
    first = run.segments[0].video
    last = run.segments[-1].video
    if abs(float(first.from_timestamp_s)) > _TIMESTAMP_EPSILON_S:
        return False
    return (
        abs(float(last.to_timestamp_s) - float(probe.duration_s))
        <= _TIMESTAMP_EPSILON_S
    )


def _run_aligned_pts(
    run: _PendingVideoRun,
    probe: _VideoSourceProbe,
) -> tuple[int, int] | None:
    if not run.segments or not probe.monotonic_pts_dts:
        return None

    start_pts = _match_timestamp_to_pts(
        float(run.segments[0].video.from_timestamp_s),
        probe.keyframe_pts,
        probe.time_base,
    )
    if start_pts is None:
        return None
    end_pts = _match_timestamp_to_pts(
        float(run.segments[-1].video.to_timestamp_s),
        probe.packet_boundary_pts,
        probe.time_base,
    )
    if end_pts is None or end_pts <= start_pts:
        return None
    return start_pts, end_pts


def run_is_remuxable(
    run: _PendingVideoRun,
    probe: _VideoSourceProbe,
) -> bool:
    if _run_is_full_source(run, probe):
        return True
    return _run_aligned_pts(run, probe) is not None


def probes_are_remux_compatible(
    left: _VideoSourceProbe,
    right: _VideoSourceProbe,
) -> bool:
    return (
        left.width == right.width
        and left.height == right.height
        and left.fps == right.fps
        and left.time_base == right.time_base
        and left.codec == right.codec
        and left.pix_fmt == right.pix_fmt
        and not left.has_audio
        and not right.has_audio
    )


def _cached_media_path(video_key: str, video: Video) -> str | None:
    if not isinstance(video.media, MediaFile):
        return None
    cached_path = submit(
        video.media.cache_file(cache_name=f"lerobot_writer:{video_key}")
    ).result()
    return str(cached_path)


@dataclass(slots=True)
class _VideoProbeCache:
    max_entries: int = _VIDEO_PROBE_CACHE_MAX_ENTRIES
    _cache: dict[tuple[str, str], _VideoSourceProbe | None] = field(
        default_factory=dict,
        init=False,
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def probe_video_source(
        self,
        *,
        video_key: str,
        video: Video,
        default_fps: int | None,
    ) -> _VideoSourceProbe | None:
        cache_key = (video_key, video.uri)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None and Path(cached.local_path).exists():
                self._cache.pop(cache_key, None)
                self._cache[cache_key] = cached
                return cached
            if cache_key in self._cache:
                self._cache.pop(cache_key, None)

        local_path = _cached_media_path(video_key, video)
        if local_path is None:
            self.cache(cache_key, None)
            return None

        with av.open(local_path, mode="r") as container:
            stream = cast(
                Any,
                next(
                    (item for item in container.streams if item.type == "video"),
                    None,
                ),
            )
            if stream is None or stream.width is None or stream.height is None:
                self.cache(cache_key, None)
                return None
            duration_s = _stream_duration_seconds(container, stream)
            if duration_s is None:
                self.cache(cache_key, None)
                return None
            stream_fps = stream.average_rate or stream.base_rate
            fps = (
                int(round(float(stream_fps)))
                if stream_fps is not None
                else int(default_fps or 30)
            )
            codec_obj = getattr(getattr(stream, "codec_context", None), "codec", None)
            codec = getattr(codec_obj, "canonical_name", None) or getattr(
                codec_obj, "name", None
            )
            pix_fmt = getattr(getattr(stream, "codec_context", None), "pix_fmt", None)
            packet_boundary_pts: list[int] = []
            keyframe_pts: list[int] = []
            monotonic_pts_dts = True
            for packet in container.demux(stream):
                if packet.pts is None or packet.dts is None:
                    continue
                packet_boundary_pts.append(int(packet.pts))
                if packet.is_keyframe:
                    keyframe_pts.append(int(packet.pts))
                if int(packet.pts) != int(packet.dts):
                    monotonic_pts_dts = False
            if stream.time_base is None:
                self.cache(cache_key, None)
                return None
            duration_pts = (
                int(stream.duration)
                if stream.duration is not None
                else int(round(duration_s / float(stream.time_base)))
            )
            packet_boundary_pts.append(duration_pts)
            probe = _VideoSourceProbe(
                local_path=local_path,
                duration_s=float(duration_s),
                width=int(stream.width),
                height=int(stream.height),
                fps=max(1, fps),
                time_base=Fraction(cast(Any, stream.time_base)),
                codec=str(codec) if codec else None,
                pix_fmt=str(pix_fmt) if pix_fmt else None,
                has_audio=any(item.type == "audio" for item in container.streams),
                packet_boundary_pts=tuple(sorted(set(packet_boundary_pts))),
                keyframe_pts=tuple(sorted(set(keyframe_pts))),
                monotonic_pts_dts=monotonic_pts_dts,
            )
        self.cache(cache_key, probe)
        return probe

    def cache(
        self,
        cache_key: tuple[str, str],
        probe: _VideoSourceProbe | None,
    ) -> None:
        with self._lock:
            self._cache.pop(cache_key, None)
            self._cache[cache_key] = probe
            while len(self._cache) > self.max_entries:
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key, None)

    def cleanup_video(self, video: Video) -> None:
        with self._lock:
            self._cache = {
                key: value for key, value in self._cache.items() if key[1] != video.uri
            }
        if isinstance(video.media, MediaFile):
            video.media.cleanup()


def probe_run_for_remux(
    *,
    run: _PendingVideoRun,
    probe_cache: _VideoProbeCache,
    default_fps: int | None,
) -> _VideoSourceProbe | None:
    if not run.segments:
        return None
    if any(
        isinstance(segment.video.media, DecodedVideo) or segment.source_stats is None
        for segment in run.segments
    ):
        return None
    probe = probe_cache.probe_video_source(
        video_key=run.video_key,
        video=run.segments[0].video,
        default_fps=default_fps,
    )
    if probe is None or not run_is_remuxable(run, probe):
        return None
    return probe


def remux_batch(
    *,
    folder: DataFolder,
    output_rel: str,
    video_key: str,
    entries: Sequence[tuple[_PendingVideoRun, _VideoSourceProbe]],
) -> None:
    if not entries:
        raise ValueError("remux requires at least one run/probe entry")

    with folder.open(output_rel, mode="wb") as output_file:
        output_container = av.open(
            output_file,
            mode="w",
            format="mp4",
            options={"movflags": _SEGMENTED_MP4_MOVFLAGS},
        )
        try:
            output_stream: Any | None = None
            output_offset_pts = 0

            for run, probe in entries:
                aligned_pts = _run_aligned_pts(run, probe)
                if aligned_pts is None:
                    raise ValueError(
                        f"Video run for {video_key!r} is not remux-aligned"
                    )
                start_pts, end_pts = aligned_pts

                with av.open(probe.local_path, mode="r") as input_container:
                    input_stream = next(
                        (
                            item
                            for item in input_container.streams
                            if item.type == "video"
                        ),
                        None,
                    )
                    if input_stream is None:
                        raise ValueError(
                            f"Video source for {video_key!r} has no video stream"
                        )
                    if output_stream is None:
                        output_stream = output_container.add_stream_from_template(
                            template=input_stream,
                            opaque=True,
                        )
                        output_stream.time_base = input_stream.time_base

                    for packet in input_container.demux(input_stream):
                        if packet.pts is None or packet.dts is None:
                            continue
                        packet_pts = int(packet.pts)
                        if packet_pts < start_pts:
                            continue
                        if packet_pts >= end_pts:
                            break
                        packet.pts = int(packet.pts) - start_pts + output_offset_pts
                        packet.dts = int(packet.dts) - start_pts + output_offset_pts
                        packet.stream = output_stream
                        output_container.mux(packet)

                output_offset_pts += end_pts - start_pts
        finally:
            output_container.close()


__all__ = [
    "_PendingRemuxBatch",
    "_PendingVideoRun",
    "_PendingVideoSegment",
    "_TIMESTAMP_EPSILON_S",
    "_VIDEO_PROBE_CACHE_MAX_ENTRIES",
    "_VideoProbeCache",
    "_VideoSourceProbe",
    "probe_run_for_remux",
    "probes_are_remux_compatible",
    "remux_batch",
    "run_can_extend",
    "run_is_remuxable",
]
