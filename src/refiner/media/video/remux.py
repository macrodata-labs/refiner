from __future__ import annotations

from dataclasses import dataclass
from typing import IO, Any

import av

from refiner.io import DataFolder
from refiner.media.video.types import VideoFile
from refiner.pipeline.utils.cache.decoder_cache import (
    OpenedVideoSource,
    VideoSourceProbe,
    get_opened_video_source_cache,
    reset_opened_video_source_cache,
)
from refiner.pipeline.utils.cache.lease_cache import CacheLease

_SEGMENTED_MP4_MOVFLAGS = "frag_keyframe+default_base_moof"


def video_from_timestamp_s(video: VideoFile) -> float:
    return float(video.from_timestamp_s or 0.0)


def video_to_timestamp_s(video: VideoFile) -> float | None:
    return video.to_timestamp_s


@dataclass(frozen=True, slots=True)
class VideoPtsAlignment:
    probe: VideoSourceProbe
    start_pts: int
    end_pts: int


@dataclass(slots=True)
class PreparedVideoSource:
    lease: CacheLease[str, OpenedVideoSource]
    video: VideoFile
    uri: str
    probe: VideoSourceProbe | None
    alignment: VideoPtsAlignment | None
    container: Any
    stream: Any

    def close(self) -> None:
        self.lease.release()


class RemuxWriter:
    def __init__(
        self,
        *,
        probe: VideoSourceProbe,
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
        probe: VideoSourceProbe,
    ) -> "RemuxWriter":
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
        return cls(probe=probe, output_file=output_file, container=container)

    @property
    def size_bytes(self) -> int:
        return int(self.output_file.tell())

    def close(self) -> None:
        self.container.close()
        self.output_file.close()

    def append_prepared_video(
        self,
        prepared: PreparedVideoSource,
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
        out_from_s = float(output_base_pts) * time_base_s
        out_to_s = float(output_base_pts + (end_pts - start_pts)) * time_base_s
        return out_from_s, out_to_s


def probes_are_remux_compatible(
    left: VideoSourceProbe,
    right: VideoSourceProbe,
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


def probe_for_remux(
    *,
    probe: VideoSourceProbe | None,
    video: VideoFile,
) -> tuple[VideoSourceProbe | None, VideoPtsAlignment | None]:
    if probe is None:
        return None, None

    video_end_s = video_to_timestamp_s(video)
    if video_end_s is None:
        return probe, None

    time_base_s = float(probe.time_base)
    start_pts = max(0, int(round(video_from_timestamp_s(video) / time_base_s)))
    end_pts = max(start_pts + 1, int(round(video_end_s / time_base_s)))
    return probe, VideoPtsAlignment(
        probe=probe,
        start_pts=start_pts,
        end_pts=end_pts,
    )


async def prepare_video_source(
    *,
    cache_key: str,
    video: VideoFile,
) -> PreparedVideoSource:
    lease = await get_opened_video_source_cache(name=cache_key).acquire(video.uri)
    source = lease.resource
    try:
        start_pts = 0
        if source.stream.time_base is not None:
            start_pts = max(
                0,
                int(
                    round(
                        video_from_timestamp_s(video) / float(source.stream.time_base)
                    )
                ),
            )

        try:
            source.container.seek(start_pts, stream=source.stream, backward=True)
        except Exception:
            try:
                source.container.seek(
                    max(0, start_pts),
                    stream=source.stream,
                    backward=True,
                    any_frame=True,
                )
            except Exception:
                pass

        probe, alignment = probe_for_remux(probe=source.probe, video=video)
        return PreparedVideoSource(
            lease=lease,
            video=video,
            uri=source.uri,
            probe=probe,
            alignment=alignment,
            container=source.container,
            stream=source.stream,
        )
    except Exception:
        lease.release()
        raise


__all__ = [
    "PreparedVideoSource",
    "RemuxWriter",
    "VideoPtsAlignment",
    "VideoSourceProbe",
    "prepare_video_source",
    "probe_for_remux",
    "probes_are_remux_compatible",
    "reset_opened_video_source_cache",
    "video_from_timestamp_s",
    "video_to_timestamp_s",
]
