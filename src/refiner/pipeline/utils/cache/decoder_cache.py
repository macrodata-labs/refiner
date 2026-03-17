from __future__ import annotations

import atexit
import asyncio
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from typing import IO, Any, cast

import av

from refiner.execution.asyncio.runtime import io_executor
from refiner.io import DataFile
from refiner.pipeline.utils.cache.lease_cache import LeaseCache


@dataclass(frozen=True, slots=True)
class VideoSourceProbe:
    width: int
    height: int
    fps: int | None
    time_base: Fraction
    codec: str | None
    pix_fmt: str | None
    has_audio: bool


@dataclass(slots=True)
class OpenedVideoSource:
    uri: str
    input_file: IO[bytes]
    container: Any
    stream: Any
    probe: VideoSourceProbe | None

    def close(self) -> None:
        try:
            self.container.close()
        finally:
            self.input_file.close()


class OpenedVideoSourceCache(LeaseCache[str, OpenedVideoSource]):
    def __init__(
        self,
        *,
        name: str,
        max_entries: int,
    ) -> None:
        super().__init__(
            max_entries=max_entries,
            max_leases_per_key=1,
            block_on_capacity=True,
        )
        self.name = name
        self.max_entries = int(max_entries)

    async def _create_resource(
        self,
        key: str,
    ) -> tuple[OpenedVideoSource, int]:
        source = await asyncio.get_running_loop().run_in_executor(
            io_executor(),
            partial(
                _open_video_source,
                uri=key,
            ),
        )
        return source, 0

    def _close_resource(self, resource: OpenedVideoSource) -> None:
        resource.close()


def _probe_video_source(
    *,
    container: Any,
) -> VideoSourceProbe | None:
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
    return VideoSourceProbe(
        width=int(stream.width),
        height=int(stream.height),
        fps=int(round(float(stream_fps))) if stream_fps is not None else None,
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


def _open_video_source(
    *,
    uri: str,
) -> OpenedVideoSource:
    input_file = DataFile.resolve(uri).open("rb")
    try:
        container = av.open(input_file, mode="r")
        stream = cast(
            Any,
            next((item for item in container.streams if item.type == "video"), None),
        )
        if stream is None:
            container.close()
            input_file.close()
            raise ValueError(f"Video source has no video stream for {uri!r}")
        probe = _probe_video_source(
            container=container,
        )
        return OpenedVideoSource(
            uri=uri,
            probe=probe,
            input_file=input_file,
            container=container,
            stream=stream,
        )
    except Exception:
        input_file.close()
        raise


_opened_video_source_cache_registry: dict[str, OpenedVideoSourceCache] = {}


def get_opened_video_source_cache(
    *,
    name: str = "default",
    max_entries: int | None = None,
) -> OpenedVideoSourceCache:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("opened video source cache name must be a non-empty string")

    normalized = name.strip()
    cache = _opened_video_source_cache_registry.get(normalized)
    if cache is None:
        cache = OpenedVideoSourceCache(
            name=normalized,
            max_entries=max_entries if max_entries is not None else 8,
        )
        _opened_video_source_cache_registry[normalized] = cache
        return cache
    if max_entries is not None and cache.max_entries != int(max_entries):
        raise ValueError(
            f"Opened video source cache {normalized!r} already exists with "
            f"max_entries={cache.max_entries}"
        )
    return cache


def reset_opened_video_source_cache(name: str | None = None) -> None:
    if name is None:
        items = list(_opened_video_source_cache_registry.values())
        _opened_video_source_cache_registry.clear()
    else:
        normalized = name.strip()
        cache = _opened_video_source_cache_registry.pop(normalized, None)
        items = [cache] if cache is not None else []

    for cache in items:
        if cache is None:
            continue
        cache.clear()


atexit.register(reset_opened_video_source_cache)


__all__ = [
    "OpenedVideoSource",
    "OpenedVideoSourceCache",
    "VideoSourceProbe",
    "get_opened_video_source_cache",
    "reset_opened_video_source_cache",
]
