from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import IO, Any, Iterator, cast

import av
from loguru import logger
from refiner.io import DataFolder
from refiner.media import Video
from refiner.pipeline.sinks.lerobot._lerobot_video_types import (
    video_from_timestamp_s,
    video_to_timestamp_s,
    video_uri,
)

_TIMESTAMP_EPSILON_S = 1e-3
_SEGMENTED_MP4_MOVFLAGS = "frag_keyframe+default_base_moof"


@dataclass(frozen=True, slots=True)
class _VideoSourceProbe:
    width: int
    height: int
    fps: int
    time_base: Fraction
    codec: str | None
    pix_fmt: str | None
    has_audio: bool


@dataclass(frozen=True, slots=True)
class _VideoPtsAlignment:
    probe: _VideoSourceProbe
    start_pts: int
    end_pts: int


@dataclass(slots=True)
class _OpenedVideoSource:
    uri: str
    input_file: IO[bytes]
    container: Any
    stream: Any


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
        self._source: _OpenedVideoSource | None = None

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
        self._close_source()
        self.container.close()
        self.output_file.close()

    def append_video(
        self,
        video: Video,
        alignment: _VideoPtsAlignment,
    ) -> tuple[float, float]:
        probe = alignment.probe
        if not probes_are_remux_compatible(self.probe, probe):
            raise ValueError("Video clip is not remux-compatible")

        start_pts = alignment.start_pts
        end_pts = alignment.end_pts
        output_base_pts = self.output_offset_pts
        time_base_s = float(probe.time_base)

        source = self._source_for(video)
        input_container = source.container
        input_stream = source.stream
        if self.stream is None:
            self.stream = self.container.add_stream_from_template(
                template=input_stream,
                opaque=True,
            )
        stream = self.stream
        if stream is None:
            raise RuntimeError("Remux output stream was not initialized")
        stream.time_base = input_stream.time_base

        try:
            input_container.seek(
                start_pts,
                stream=input_stream,
                backward=True,
            )
        except Exception:
            try:
                input_container.seek(
                    max(0, start_pts),
                    stream=input_stream,
                    backward=True,
                    any_frame=True,
                )
            except Exception:
                pass

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
        out_from_s = float(output_base_pts) * time_base_s
        out_to_s = float(output_base_pts + (end_pts - start_pts)) * time_base_s
        return out_from_s, out_to_s

    def _source_for(self, video: Video) -> _OpenedVideoSource:
        source_uri = video_uri(video)
        current = self._source
        if current is not None and current.uri == source_uri:
            return current

        if current is not None:
            logger.debug(
                "Invalidating remux source container: old_uri={!r} new_uri={!r}",
                current.uri,
                source_uri,
            )
        self._close_source()
        input_file = video.open("rb")
        try:
            container = av.open(input_file, mode="r")
            input_stream = next(
                (item for item in container.streams if item.type == "video"),
                None,
            )
            if input_stream is None:
                raise ValueError("Video source has no video stream")
            opened = _OpenedVideoSource(
                uri=source_uri,
                input_file=input_file,
                container=container,
                stream=input_stream,
            )
            logger.debug(
                "Opened remux source container: uri={!r}",
                source_uri,
            )
            self._source = opened
            return opened
        except Exception:
            input_file.close()
            raise

    def _close_source(self) -> None:
        source = self._source
        if source is None:
            return
        logger.debug(
            "Closing remux source container: uri={!r}",
            source.uri,
        )
        try:
            source.container.close()
        finally:
            source.input_file.close()
            self._source = None


@contextmanager
def _open_video_container(video: Video) -> Iterator[Any]:
    input_file = video.open("rb")
    try:
        with av.open(input_file, mode="r") as container:
            yield container
    finally:
        input_file.close()


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


def _probe_video_source(
    *,
    video: Video,
    default_fps: int | None,
) -> _VideoSourceProbe | None:
    with _open_video_container(video) as container:
        stream = cast(
            Any,
            next((item for item in container.streams if item.type == "video"), None),
        )
        if stream is None or stream.width is None or stream.height is None:
            return None
        if stream.time_base is None:
            return None

        stream_fps = stream.average_rate or stream.base_rate
        codec_obj = getattr(getattr(stream, "codec_context", None), "codec", None)
        return _VideoSourceProbe(
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
                str(getattr(getattr(stream, "codec_context", None), "pix_fmt", None))
                if getattr(getattr(stream, "codec_context", None), "pix_fmt", None)
                else None
            ),
            has_audio=any(item.type == "audio" for item in container.streams),
        )


async def probe_video_for_remux(
    *,
    video: Video,
    source_stats: dict[str, Any] | None,
    default_fps: int | None,
) -> _VideoPtsAlignment | None:
    if source_stats is None:
        return None

    probe = await asyncio.to_thread(
        _probe_video_source,
        video=video,
        default_fps=default_fps,
    )
    if probe is None:
        return None
    time_base_s = float(probe.time_base)
    start_pts = max(
        0,
        int(round(video_from_timestamp_s(video) / time_base_s)),
    )
    video_end_s = video_to_timestamp_s(video)
    if video_end_s is None:
        return None
    end_pts = max(
        start_pts + 1,
        int(round(video_end_s / time_base_s)),
    )
    return _VideoPtsAlignment(
        probe=probe,
        start_pts=start_pts,
        end_pts=end_pts,
    )
