from __future__ import annotations

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


class RemuxWriter:
    def __init__(
        self,
        *,
        video_key: str,
        file_idx: int,
        probe: _VideoSourceProbe,
        output_file: IO[bytes],
        container: Any,
    ) -> None:
        self.video_key = video_key
        self.file_idx = file_idx
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
        video_key: str,
        file_idx: int,
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
            video_key=video_key,
            file_idx=file_idx,
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


def _run_aligned_pts(
    run: _PendingVideoRun,
    probe: _VideoSourceProbe,
) -> tuple[int, int] | None:
    if not run.segments or not probe.monotonic_pts_dts:
        return None

    start_pts = _match_timestamp_to_pts(
        video_from_timestamp_s(run.segments[0].video),
        probe.keyframe_pts,
        probe.time_base,
    )
    if start_pts is None:
        return None
    run_end_s = video_to_timestamp_s(run.segments[-1].video)
    if run_end_s is None:
        return None
    end_pts = _match_timestamp_to_pts(
        run_end_s,
        probe.packet_boundary_pts,
        probe.time_base,
    )
    if end_pts is None or end_pts <= start_pts:
        return None
    return start_pts, end_pts


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
) -> _VideoSourceProbe | None:
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

    first = run.segments[0].video
    last = run.segments[-1].video
    last_to = video_to_timestamp_s(last)
    if (
        abs(video_from_timestamp_s(first)) <= _TIMESTAMP_EPSILON_S
        and last_to is not None
        and abs(last_to - float(probe.duration_s)) <= _TIMESTAMP_EPSILON_S
    ):
        return probe
    return probe if _run_aligned_pts(run, probe) is not None else None


def append_remux_run(
    *,
    writer: RemuxWriter,
    run: _PendingVideoRun,
    probe: _VideoSourceProbe,
) -> None:
    if not probes_are_remux_compatible(writer.probe, probe):
        raise ValueError(f"Video run for {writer.video_key!r} is not remux-compatible")

    aligned_pts = _run_aligned_pts(run, probe)
    if aligned_pts is None:
        raise ValueError(f"Video run for {writer.video_key!r} is not remux-aligned")
    start_pts, end_pts = aligned_pts

    with av.open(probe.local_path, mode="r") as input_container:
        input_stream = next(
            (item for item in input_container.streams if item.type == "video"),
            None,
        )
        if input_stream is None:
            raise ValueError(
                f"Video source for {writer.video_key!r} has no video stream"
            )
        if writer.stream is None:
            writer.stream = writer.container.add_stream_from_template(
                template=input_stream,
                opaque=True,
            )
            writer.stream.time_base = input_stream.time_base

        for packet in input_container.demux(input_stream):
            if packet.pts is None or packet.dts is None:
                continue
            packet_pts = int(packet.pts)
            if packet_pts < start_pts:
                continue
            if packet_pts >= end_pts:
                break
            packet.pts = int(packet.pts) - start_pts + writer.output_offset_pts
            packet.dts = int(packet.dts) - start_pts + writer.output_offset_pts
            packet.stream = writer.stream
            writer.container.mux(packet)

    writer.output_offset_pts += end_pts - start_pts
    run_end_s = video_to_timestamp_s(run.segments[-1].video)
    if run_end_s is None:
        raise ValueError(f"Video run for {writer.video_key!r} is missing an end time")
    writer.duration_s += run_end_s - video_from_timestamp_s(run.segments[0].video)
