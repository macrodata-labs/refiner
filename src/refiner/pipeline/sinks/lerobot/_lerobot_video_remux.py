from __future__ import annotations

from bisect import bisect_left
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import IO, Any, cast

import av

from refiner.execution.asyncio.runtime import submit
from refiner.io import DataFolder
from refiner.media import Video
from refiner.media.video.types import DecodedVideo, VideoFile
from refiner.pipeline.sinks.lerobot._lerobot_video import (
    _PendingVideoRun,
    _PendingVideoSegment,
    run_can_extend,
    video_from_timestamp_s,
    video_to_timestamp_s,
    video_uri,
)

_ = (_PendingVideoSegment, run_can_extend)

_TIMESTAMP_EPSILON_S = 1e-3
_VIDEO_PROBE_CACHE_MAX_ENTRIES = 256
_SEGMENTED_MP4_MOVFLAGS = "frag_keyframe+default_base_moof"


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


@dataclass(frozen=True, slots=True)
class _RunPtsAlignment:
    probe: _VideoSourceProbe
    start_pts: int
    end_pts: int
    segment_end_pts: tuple[int, ...]


class RemuxWriter:
    def __init__(
        self,
        *,
        probe: _VideoSourceProbe,
        output_file: IO[bytes],
        container: Any,
    ) -> None:
        self.probe = probe
        self.output_file = output_file
        self.container = container
        self.stream: Any | None = None
        self.output_offset_pts = 0
        self.duration_s = 0.0

    @classmethod
    def open(
        cls,
        *,
        folder: DataFolder,
        output_rel: str,
        probe: _VideoSourceProbe,
    ) -> RemuxWriter:
        output_abs = folder._join(output_rel)
        folder.fs.makedirs(folder.fs._parent(output_abs), exist_ok=True)
        output_file = folder.open(output_rel, mode="wb")
        try:
            container = av.open(
                output_file,
                mode="w",
                format="mp4",
                options={"movflags": _SEGMENTED_MP4_MOVFLAGS},
            )
        except Exception:
            output_file.close()
            raise
        return cls(
            probe=probe,
            output_file=output_file,
            container=container,
        )

    @property
    def size_bytes(self) -> int:
        return int(self.output_file.tell())

    def close(self) -> None:
        self.container.close()
        self.output_file.close()

    def add_run(
        self,
        run: _PendingVideoRun,
        alignment: _RunPtsAlignment,
    ) -> list[tuple[float, float]]:
        probe = alignment.probe
        if not probes_are_remux_compatible(self.probe, probe):
            raise ValueError("Video run is not remux-compatible")

        start_pts = alignment.start_pts
        end_pts = alignment.end_pts
        segment_end_pts = list(alignment.segment_end_pts)
        segment_start_pts = [start_pts, *segment_end_pts[:-1]]
        output_base_pts = self.output_offset_pts
        time_base_s = float(probe.time_base)

        with av.open(probe.local_path, mode="r") as input_container:
            input_stream = next(
                (item for item in input_container.streams if item.type == "video"),
                None,
            )
            if input_stream is None:
                raise ValueError("Video source has no video stream")
            if self.stream is None:
                self.stream = self.container.add_stream_from_template(
                    template=input_stream,
                    opaque=True,
                )
            stream = self.stream
            if stream is None:
                raise RuntimeError("Remux output stream was not initialized")
            stream.time_base = input_stream.time_base

            for packet in input_container.demux(input_stream):
                if packet.pts is None or packet.dts is None:
                    continue
                packet_pts = int(packet.pts)
                if packet_pts < start_pts:
                    continue
                if packet_pts >= end_pts:
                    break
                packet.pts = int(packet.pts) - start_pts + self.output_offset_pts
                packet.dts = int(packet.dts) - start_pts + self.output_offset_pts
                packet.stream = stream
                self.container.mux(packet)

        self.output_offset_pts += end_pts - start_pts
        self.duration_s = float(self.output_offset_pts) * time_base_s
        flushed_timestamps: list[tuple[float, float]] = []
        for index in range(len(run.segments)):
            segment_start = segment_start_pts[index]
            segment_end = segment_end_pts[index]
            out_from_s = (
                float(output_base_pts + (segment_start - start_pts)) * time_base_s
            )
            out_to_s = float(output_base_pts + (segment_end - start_pts)) * time_base_s
            flushed_timestamps.append((out_from_s, out_to_s))
        return flushed_timestamps


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
    *,
    min_index: int = 0,
) -> tuple[int, int] | None:
    if min_index < 0:
        min_index = 0
    if min_index >= len(candidates):
        return None

    time_base_f = float(time_base)
    target_pts = float(timestamp_s) / time_base_f
    insert_idx = bisect_left(candidates, int(round(target_pts)), lo=min_index)
    search_indices = [insert_idx]
    if insert_idx - 1 >= min_index:
        search_indices.append(insert_idx - 1)

    best: tuple[int, int, float] | None = None
    for idx in search_indices:
        if idx < min_index or idx >= len(candidates):
            continue
        candidate = int(candidates[idx])
        error_s = abs((float(candidate) * time_base_f) - float(timestamp_s))
        if error_s > _TIMESTAMP_EPSILON_S:
            continue
        if best is None or error_s < best[2]:
            best = (candidate, idx, error_s)
    if best is None:
        return None
    return best[0], best[1]


def _run_aligned_pts(
    run: _PendingVideoRun,
    probe: _VideoSourceProbe,
) -> tuple[int, int, int] | None:
    if not run.segments or not probe.monotonic_pts_dts:
        return None

    start_match = _match_timestamp_to_pts(
        video_from_timestamp_s(run.segments[0].video),
        probe.keyframe_pts,
        probe.time_base,
    )
    if start_match is None:
        return None
    start_pts, _ = start_match
    run_end_s = video_to_timestamp_s(run.segments[-1].video)
    if run_end_s is None:
        return None
    end_match = _match_timestamp_to_pts(
        run_end_s,
        probe.packet_boundary_pts,
        probe.time_base,
    )
    if end_match is None:
        return None
    end_pts, end_idx = end_match
    if end_pts <= start_pts:
        return None
    return start_pts, end_pts, end_idx


def _run_segment_end_pts(
    *,
    run: _PendingVideoRun,
    probe: _VideoSourceProbe,
    start_pts: int,
    end_pts: int,
    end_pts_index: int,
) -> list[int]:
    segment_end_pts: list[int] = []
    previous_end = start_pts
    search_min_index = 0
    for index, segment in enumerate(run.segments):
        segment_to_s = video_to_timestamp_s(segment.video)
        if index == len(run.segments) - 1:
            segment_end = end_pts
        else:
            if segment_to_s is None:
                segment_to_s = video_from_timestamp_s(run.segments[index + 1].video)
            resolved = _match_timestamp_to_pts(
                timestamp_s=float(segment_to_s),
                candidates=probe.packet_boundary_pts,
                time_base=probe.time_base,
                min_index=search_min_index,
            )
            if resolved is None:
                raise ValueError(
                    f"Video run for {run.video_key!r} has non-aligned segment boundary"
                )
            segment_end, resolved_index = resolved
            if resolved_index > end_pts_index:
                raise ValueError(
                    f"Video run for {run.video_key!r} has non-aligned segment boundary"
                )
            search_min_index = resolved_index

        if segment_end < previous_end or segment_end > end_pts:
            raise ValueError(
                f"Video run for {run.video_key!r} has invalid segment layout"
            )
        segment_end_pts.append(segment_end)
        previous_end = segment_end
    return segment_end_pts


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
        cache_key = (video_key, video_uri(video))
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None and Path(cached.local_path).exists():
                self._cache.pop(cache_key, None)
                self._cache[cache_key] = cached
                return cached
            self._cache.pop(cache_key, None)

        if not isinstance(video, VideoFile):
            self.cache(cache_key, None)
            return None
        local_path = str(
            submit(video.cache_file(cache_name=f"lerobot_writer:{video_key}")).result()
        )

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
            if duration_s is None or stream.time_base is None:
                self.cache(cache_key, None)
                return None

            packet_boundary_pts: list[int] = []
            keyframe_pts: list[int] = []
            monotonic_pts_dts = True
            for packet in container.demux(stream):
                if packet.pts is None or packet.dts is None:
                    continue
                packet_boundary_pts.append(int(packet.pts))
                if packet.is_keyframe:
                    keyframe_pts.append(int(packet.pts))
                monotonic_pts_dts &= int(packet.pts) == int(packet.dts)

            stream_fps = stream.average_rate or stream.base_rate
            codec_obj = getattr(getattr(stream, "codec_context", None), "codec", None)
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
                fps=max(
                    1,
                    int(round(float(stream_fps)))
                    if stream_fps is not None
                    else int(default_fps or 30),
                ),
                time_base=Fraction(cast(Any, stream.time_base)),
                codec=str(
                    getattr(codec_obj, "canonical_name", None)
                    or getattr(codec_obj, "name", None)
                )
                if codec_obj is not None
                else None,
                pix_fmt=(
                    str(
                        getattr(getattr(stream, "codec_context", None), "pix_fmt", None)
                    )
                    if getattr(getattr(stream, "codec_context", None), "pix_fmt", None)
                    else None
                ),
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
                self._cache.pop(next(iter(self._cache)), None)

    def cleanup_video(self, video: Video) -> None:
        uri = video_uri(video)
        with self._lock:
            for key in [key for key in self._cache if key[1] == uri]:
                self._cache.pop(key, None)
        if isinstance(video, VideoFile):
            video.cleanup()
        elif isinstance(video, DecodedVideo):
            video.original_file.cleanup()


def probe_run_for_remux(
    *,
    run: _PendingVideoRun,
    probe_cache: _VideoProbeCache,
    default_fps: int | None,
) -> _RunPtsAlignment | None:
    if not run.segments:
        return None
    if any(
        isinstance(segment.video, DecodedVideo) or segment.source_stats is None
        for segment in run.segments
    ):
        return None

    probe = probe_cache.probe_video_source(
        video_key=run.video_key,
        video=run.segments[0].video,
        default_fps=default_fps,
    )
    if probe is None:
        return None
    if not probe.monotonic_pts_dts:
        return None

    start_match = _run_aligned_pts(run, probe)
    if start_match is None:
        return None
    start_pts, end_pts, end_pts_index = start_match

    try:
        segment_end_pts = _run_segment_end_pts(
            run=run,
            probe=probe,
            start_pts=start_pts,
            end_pts=end_pts,
            end_pts_index=end_pts_index,
        )
    except ValueError:
        return None
    return _RunPtsAlignment(
        probe=probe,
        start_pts=start_pts,
        end_pts=end_pts,
        segment_end_pts=tuple(segment_end_pts),
    )
