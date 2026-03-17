from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import partial
from fractions import Fraction
import threading
from typing import IO, Any, cast

import av
from loguru import logger
from refiner.execution.asyncio.runtime import io as io_executor
from refiner.io import DataFolder
from refiner.media import Video
from refiner.pipeline.sinks.lerobot._lerobot_video_types import (
    video_from_timestamp_s,
    video_to_timestamp_s,
    video_uri,
)

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
class _PreparedSource:
    uri: str
    probe: _VideoSourceProbe | None
    alignment: _VideoPtsAlignment | None
    input_file: IO[bytes]
    container: Any
    stream: Any

    def close(self) -> None:
        try:
            logger.debug(
                "Closing prepared remux source: uri={!r}",
                self.uri,
            )
            self.container.close()
        finally:
            self.input_file.close()


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
        logger.debug(
            "Opened remux output container: output_rel={!r}",
            output_rel,
        )
        return cls(
            probe=probe,
            output_file=output_file,
            container=container,
        )

    @property
    def size_bytes(self) -> int:
        return int(self.output_file.tell())

    def close(self) -> None:
        logger.debug("Closing remux output container")
        self.container.close()
        self.output_file.close()

    def append_prepared_video(
        self,
        prepared: _PreparedSource,
    ) -> tuple[float, float]:
        if (
            prepared.probe is None
            or prepared.alignment is None
            or not probes_are_remux_compatible(prepared.probe, self.probe)
        ):
            raise ValueError("Prepared source is not remuxable")
        probe = prepared.probe
        input_container = prepared.container
        input_stream = prepared.stream
        start_pts = prepared.alignment.start_pts
        end_pts = prepared.alignment.end_pts

        output_base_pts = self.output_offset_pts
        time_base_s = float(probe.time_base)
        packets_muxed = 0
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
            packets_muxed += 1

        self.output_offset_pts += end_pts - start_pts
        out_from_s = float(output_base_pts) * time_base_s
        out_to_s = float(output_base_pts + (end_pts - start_pts)) * time_base_s
        logger.debug(
            "Finished remux append: packets_muxed={} out_from_s={:.6f} out_to_s={:.6f}",
            packets_muxed,
            out_from_s,
            out_to_s,
        )
        return out_from_s, out_to_s


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
    container: Any,
    default_fps: int | None,
) -> _VideoSourceProbe | None:
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


def probe_for_remux(
    *,
    container: Any,
    video: Video,
    default_fps: int | None,
) -> tuple[_VideoSourceProbe | None, _VideoPtsAlignment | None]:
    probe = _probe_video_source(
        container=container,
        default_fps=default_fps,
    )
    if probe is None:
        return None, None

    video_end_s = video_to_timestamp_s(video)
    if video_end_s is None:
        return probe, None

    time_base_s = float(probe.time_base)
    start_pts = max(
        0,
        int(round(video_from_timestamp_s(video) / time_base_s)),
    )
    end_pts = max(
        start_pts + 1,
        int(round(video_end_s / time_base_s)),
    )
    return probe, _VideoPtsAlignment(
        probe=probe,
        start_pts=start_pts,
        end_pts=end_pts,
    )


def _prepare_video(
    *,
    video: Video,
    default_fps: int | None,
) -> _PreparedSource | None:
    input_file = video.open("rb")
    try:
        logger.debug(
            "Opening remux source during prepare: uri={!r}",
            video_uri(video),
        )
        container = av.open(input_file, mode="r")
        stream = cast(
            Any,
            next((item for item in container.streams if item.type == "video"), None),
        )
        if stream is None:
            container.close()
            input_file.close()
            return None

        start_pts = 0
        if stream.time_base is not None:
            start_pts = max(
                0,
                int(round(video_from_timestamp_s(video) / float(stream.time_base))),
            )

        try:
            logger.debug(
                "Preparing remux source seek: uri={!r} start_pts={} thread_id={} backward=True any_frame=False",
                video_uri(video),
                start_pts,
                threading.get_ident(),
            )
            container.seek(
                start_pts,
                stream=stream,
                backward=True,
            )
        except Exception:
            try:
                logger.debug(
                    "Retrying prepare remux seek: uri={!r} start_pts={} backward=True any_frame=True",
                    video_uri(video),
                    max(0, start_pts),
                )
                container.seek(
                    max(0, start_pts),
                    stream=stream,
                    backward=True,
                    any_frame=True,
                )
            except Exception:
                pass

        probe, alignment = probe_for_remux(
            container=container,
            video=video,
            default_fps=default_fps,
        )

        logger.debug(
            "Prepared video source: uri={!r} start_pts={} end_pts={} remuxable={}",
            video_uri(video),
            start_pts,
            alignment.end_pts if alignment is not None else None,
            alignment is not None,
        )
        return _PreparedSource(
            uri=video_uri(video),
            probe=probe,
            alignment=alignment,
            input_file=input_file,
            container=container,
            stream=stream,
        )
    except Exception:
        input_file.close()
        raise


async def prepare_video(
    *,
    video: Video,
    default_fps: int | None,
) -> _PreparedSource | None:

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        io_executor,
        partial(_prepare_video, video=video, default_fps=default_fps),
    )
