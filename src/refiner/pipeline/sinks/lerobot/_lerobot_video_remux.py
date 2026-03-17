from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from typing import IO, Any, cast

import av
from loguru import logger
from refiner.execution.asyncio.runtime import io_executor
from refiner.io import DataFolder
from refiner.media import Video
from refiner.pipeline.sinks.lerobot._lerobot_video_types import (
    video_from_timestamp_s,
    video_to_timestamp_s,
    video_uri,
)

_SEGMENTED_MP4_MOVFLAGS = "frag_keyframe+default_base_moof"


@dataclass(slots=True)
class _OpenedVideoSource:
    uri: str
    input_file: IO[bytes]
    container: Any
    stream: Any
    probe: _VideoSourceProbe | None

    def close(self) -> None:
        try:
            self.container.close()
        finally:
            self.input_file.close()


@dataclass(slots=True)
class _OpenedVideoSourceEntry:
    ready: asyncio.Future[None] | None = None
    available: asyncio.Event | None = None
    source: _OpenedVideoSource | None = None
    error: BaseException | None = None
    ref_count: int = 0


class _OpenedVideoSourceLease:
    def __init__(
        self,
        *,
        cache: "_OpenedVideoSourceCache",
        uri: str,
        source: _OpenedVideoSource,
    ) -> None:
        self._cache = cache
        self._uri = uri
        self.source = source
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._cache.release(self._uri)


class _OpenedVideoSourceCache:
    def __init__(
        self,
        *,
        video_key: str,
        default_fps: int | None,
        max_entries: int = 8,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        self.video_key = video_key
        self.default_fps = default_fps
        self.max_entries = int(max_entries)
        self.lock = asyncio.Lock()
        self._entries: OrderedDict[str, _OpenedVideoSourceEntry] = OrderedDict()

    async def acquire(
        self,
        *,
        video: Video,
    ) -> _OpenedVideoSourceLease:
        uri = video_uri(video)
        while True:
            entry = self._entries.get(uri)
            if entry is None:
                logger.debug(
                    "opened video source cache miss: video_key={!r} uri={!r}",
                    self.video_key,
                    uri,
                )
                entry = _OpenedVideoSourceEntry(
                    ready=asyncio.get_running_loop().create_future(),
                    available=asyncio.Event(),
                )
                self._entries[uri] = entry
                self._entries.move_to_end(uri)
                break
            if entry.error is not None:
                logger.debug(
                    "opened video source cache load error: video_key={!r} uri={!r}",
                    self.video_key,
                    uri,
                )
                self._entries.pop(uri, None)
                raise entry.error
            if entry.source is not None and entry.ref_count == 0:
                logger.debug(
                    "opened video source cache hit: video_key={!r} uri={!r}",
                    self.video_key,
                    uri,
                )
                entry.ref_count = 1
                if entry.available is not None:
                    entry.available.clear()
                self._entries.move_to_end(uri)
                return _OpenedVideoSourceLease(
                    cache=self,
                    uri=uri,
                    source=entry.source,
                )
            if entry.source is None:
                if entry.ready is None:
                    raise RuntimeError(f"Opened source state is invalid for {uri!r}")
                logger.debug(
                    "opened video source cache wait for create: video_key={!r} uri={!r}",
                    self.video_key,
                    uri,
                )
                await entry.ready
            else:
                if entry.available is None:
                    raise RuntimeError(f"Opened source state is invalid for {uri!r}")
                logger.debug(
                    "opened video source cache wait for lease: video_key={!r} uri={!r}",
                    self.video_key,
                    uri,
                )
                await entry.available.wait()

        try:
            logger.debug(
                "opened video source cache creating source: video_key={!r} uri={!r}",
                self.video_key,
                uri,
            )
            source = await asyncio.get_running_loop().run_in_executor(
                io_executor(),
                partial(_open_video_source, video=video, default_fps=self.default_fps),
            )
        except BaseException as exc:
            entry = self._entries.get(uri)
            if entry is not None and entry.source is None:
                entry.error = exc
                if entry.ready is not None and not entry.ready.done():
                    entry.ready.set_result(None)
            logger.debug(
                "opened video source cache create failed: video_key={!r} uri={!r} error={!r}",
                self.video_key,
                uri,
                exc,
            )
            raise

        entry = self._entries.get(uri)
        if entry is None or entry.source is not None:
            source.close()
            return await self.acquire(video=video)

        entry.source = source
        entry.error = None
        entry.ref_count = 1
        if entry.available is None:
            entry.available = asyncio.Event()
        entry.available.clear()
        if entry.ready is not None and not entry.ready.done():
            entry.ready.set_result(None)
        self._entries.move_to_end(uri)
        logger.debug(
            "opened video source cache created source: video_key={!r} uri={!r}",
            self.video_key,
            uri,
        )
        for item in self._evict_idle():
            item.close()
        return _OpenedVideoSourceLease(
            cache=self,
            uri=uri,
            source=source,
        )

    def release(self, uri: str) -> None:
        entry = self._entries.get(uri)
        if entry is None or entry.source is None:
            return
        entry.ref_count = max(0, entry.ref_count - 1)
        logger.debug(
            "opened video source cache release: video_key={!r} uri={!r} ref_count={}",
            self.video_key,
            uri,
            entry.ref_count,
        )
        if entry.ref_count == 0 and entry.available is not None:
            entry.available.set()
            for item in self._evict_idle():
                item.close()

    def clear(self) -> None:
        to_close = [
            entry.source for entry in self._entries.values() if entry.source is not None
        ]
        if to_close:
            logger.debug(
                "opened video source cache clear: video_key={!r} entries={}",
                self.video_key,
                len(to_close),
            )
        self._entries.clear()
        for item in to_close:
            item.close()

    def _evict_idle(self) -> list[_OpenedVideoSource]:
        to_close: list[_OpenedVideoSource] = []
        while self._ready_entry_count() > self.max_entries:
            candidate_key = next(
                (
                    key
                    for key, entry in self._entries.items()
                    if entry.source is not None and entry.ref_count == 0
                ),
                None,
            )
            if candidate_key is None:
                break
            entry = self._entries.pop(candidate_key)
            if entry.source is not None:
                logger.debug(
                    "opened video source cache evict idle source: video_key={!r} uri={!r}",
                    self.video_key,
                    candidate_key,
                )
                to_close.append(entry.source)
        return to_close

    def _ready_entry_count(self) -> int:
        return sum(1 for entry in self._entries.values() if entry.source is not None)


_opened_video_source_cache_registry: dict[str, _OpenedVideoSourceCache] = {}


def _get_opened_video_source_cache(
    *,
    video_key: str,
    default_fps: int | None,
) -> _OpenedVideoSourceCache:
    cache = _opened_video_source_cache_registry.get(video_key)
    if cache is None:
        cache = _OpenedVideoSourceCache(
            video_key=video_key,
            default_fps=default_fps,
        )
        _opened_video_source_cache_registry[video_key] = cache
        return cache
    if cache.default_fps != default_fps:
        raise ValueError(
            f"Opened video source cache for {video_key!r} already exists with "
            f"default_fps={cache.default_fps!r}"
        )
    return cache


def reset_opened_video_source_cache(video_key: str | None = None) -> None:
    if video_key is None:
        caches = list(_opened_video_source_cache_registry.values())
        _opened_video_source_cache_registry.clear()
    else:
        cache = _opened_video_source_cache_registry.pop(video_key, None)
        caches = [cache] if cache is not None else []

    for cache in caches:
        if cache is None:
            continue
        cache.clear()


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
    lease: _OpenedVideoSourceLease
    uri: str
    probe: _VideoSourceProbe | None
    alignment: _VideoPtsAlignment | None
    container: Any
    stream: Any

    def close(self) -> None:
        self.lease.release()


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
    probe: _VideoSourceProbe | None,
    video: Video,
) -> tuple[_VideoSourceProbe | None, _VideoPtsAlignment | None]:
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


def _open_video_source(
    *,
    video: Video,
    default_fps: int | None,
) -> _OpenedVideoSource:
    input_file = video.open("rb")
    try:
        container = av.open(input_file, mode="r")
        stream = cast(
            Any,
            next((item for item in container.streams if item.type == "video"), None),
        )
        if stream is None:
            container.close()
            input_file.close()
            raise ValueError(f"Video source has no video stream for {video.uri!r}")
        probe = _probe_video_source(
            container=container,
            default_fps=default_fps,
        )
        return _OpenedVideoSource(
            uri=video_uri(video),
            probe=probe,
            input_file=input_file,
            container=container,
            stream=stream,
        )
    except Exception:
        input_file.close()
        raise


async def prepare_video(
    *,
    video_key: str,
    video: Video,
    default_fps: int | None,
) -> _PreparedSource:
    lease = await _get_opened_video_source_cache(
        video_key=video_key,
        default_fps=default_fps,
    ).acquire(video=video)
    source = lease.source
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
            source.container.seek(
                start_pts,
                stream=source.stream,
                backward=True,
            )
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

        probe, alignment = probe_for_remux(
            probe=source.probe,
            video=video,
        )
        return _PreparedSource(
            lease=lease,
            uri=source.uri,
            probe=probe,
            alignment=alignment,
            container=source.container,
            stream=source.stream,
        )
    except Exception:
        lease.release()
        raise
