from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from refiner.io import DataFile
from refiner.pipeline.utils.cache.file_cache import _CacheFileLease, MediaLocalCache
from refiner.pipeline.utils.cache.lease_cache import CacheLease, LeaseCache

if TYPE_CHECKING:
    import av


@dataclass(slots=True)
class _DecoderResource:
    container: Any
    stream_index: int
    decode_lock: asyncio.Lock
    file_lease: _CacheFileLease | None = None
    configured_decoder_threads: int | None = None
    last_collect_window: tuple[float, float | None] | None = None
    last_collect_result: tuple[tuple[np.ndarray, ...], DecodeWindowMeta] | None = None


@dataclass(frozen=True, slots=True)
class DecodeWindowMeta:
    frame_count: int
    width: int | None
    height: int | None
    pix_fmt: str | None


class _DecoderLeaseCache(LeaseCache[DataFile, _DecoderResource]):
    def __init__(
        self,
        *,
        name: str,
        max_decoders: int,
        media_cache: MediaLocalCache,
        av_module: Any,
    ) -> None:
        super().__init__(max_entries=max_decoders)
        self.name = name
        self._av = av_module
        self._media_cache = media_cache

    async def _create_resource(
        self,
        key: DataFile,
    ) -> tuple[_DecoderResource, int]:
        # Check if we have a valid lease of the media file
        file_lease = await self._media_cache.acquire_file_lease(key)
        try:
            container = self._av.open(file_lease.path)
            stream = container.streams.video[0]
            return (
                _DecoderResource(
                    container=container,
                    stream_index=int(stream.index),
                    decode_lock=asyncio.Lock(),
                    file_lease=file_lease,
                ),
                1,
            )
        except Exception:
            file_lease.release()
            raise

    def _close_resource(self, resource: _DecoderResource) -> None:
        resource.container.close()
        if resource.file_lease is not None:
            resource.file_lease.release()


class DecoderLease:
    def __init__(
        self,
        *,
        cache_lease: CacheLease[DataFile, _DecoderResource],
        owner: "VideoDecoderCache",
    ) -> None:
        self._cache_lease = cache_lease
        self._owner = owner
        self._released = False

    async def decode_window(
        self,
        *,
        from_timestamp_s: float,
        to_timestamp_s: float | None,
        on_frame: Callable[["av.VideoFrame"], None],
        decoder_threads: int | None = None,
    ) -> DecodeWindowMeta:
        start_ts, end_ts = self._owner._validate_bounds(
            from_timestamp_s=from_timestamp_s,
            to_timestamp_s=to_timestamp_s,
        )
        resource = self._cache_lease.resource
        async with resource.decode_lock:
            self._owner._configure_decoder_threads(
                resource=resource,
                decoder_threads=decoder_threads,
            )
            return self._owner._decode_with_decoder(
                container=resource.container,
                stream_index=resource.stream_index,
                from_timestamp_s=start_ts,
                to_timestamp_s=end_ts,
                on_frame=on_frame,
            )

    async def decode_window_collect_rgb24(
        self,
        *,
        from_timestamp_s: float,
        to_timestamp_s: float | None,
        decoder_threads: int | None = None,
    ) -> tuple[tuple[np.ndarray, ...], DecodeWindowMeta]:
        start_ts, end_ts = self._owner._validate_bounds(
            from_timestamp_s=from_timestamp_s,
            to_timestamp_s=to_timestamp_s,
        )
        resource = self._cache_lease.resource
        async with resource.decode_lock:
            self._owner._configure_decoder_threads(
                resource=resource,
                decoder_threads=decoder_threads,
            )
            cached = resource.last_collect_result
            if (
                resource.last_collect_window == (start_ts, end_ts)
                and cached is not None
            ):
                return cached

            frames: list[np.ndarray] = []
            meta = self._owner._decode_with_decoder(
                container=resource.container,
                stream_index=resource.stream_index,
                from_timestamp_s=start_ts,
                to_timestamp_s=end_ts,
                on_frame=lambda frame: frames.append(frame.to_ndarray(format="rgb24")),
            )
            frames_tuple = tuple(frames)
            resource.last_collect_window = (start_ts, end_ts)
            resource.last_collect_result = (frames_tuple, meta)
            return frames_tuple, meta

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._cache_lease.release()

    async def probe_fps(self) -> int | None:
        resource = self._cache_lease.resource
        async with resource.decode_lock:
            input_stream = resource.container.streams.video[resource.stream_index]
            for rate in (
                input_stream.average_rate,
                input_stream.guessed_rate,
                input_stream.base_rate,
            ):
                if rate is None:
                    continue
                return int(round(float(rate)))
            return None


class VideoDecoderCache:
    """Named in-memory cache for decoded AV containers."""

    def __init__(
        self,
        *,
        name: str,
        max_decoders: int = 16,
        media_cache: MediaLocalCache,
    ) -> None:
        if max_decoders <= 0:
            raise ValueError("max_decoders must be > 0")

        import av

        self.name = name
        self.max_decoders = int(max_decoders)
        self._cache = _DecoderLeaseCache(
            name=name,
            max_decoders=self.max_decoders,
            media_cache=media_cache,
            av_module=av,
        )

    async def acquire(
        self,
        *,
        data_file: DataFile,
    ) -> DecoderLease:
        cache_lease = await self._cache.acquire(data_file)
        return DecoderLease(cache_lease=cache_lease, owner=self)

    async def decode_segment(
        self,
        *,
        data_file: DataFile,
        from_timestamp_s: float,
        to_timestamp_s: float | None,
        decoder_threads: int | None = None,
    ) -> tuple[tuple[np.ndarray, ...], int | None, int | None, str | None]:
        lease = await self.acquire(data_file=data_file)
        try:
            frames_tuple, meta = await lease.decode_window_collect_rgb24(
                from_timestamp_s=from_timestamp_s,
                to_timestamp_s=to_timestamp_s,
                decoder_threads=decoder_threads,
            )
        finally:
            lease.release()

        if meta.frame_count <= 0:
            raise ValueError(
                "Video segment contains no decodable frames in requested timestamp window."
            )
        return frames_tuple, meta.width, meta.height, meta.pix_fmt

    async def decode_segment_with_callback_from_data_file(
        self,
        *,
        data_file: DataFile,
        from_timestamp_s: float,
        to_timestamp_s: float | None,
        on_frame: Callable[["av.VideoFrame"], None],
        decoder_threads: int | None = None,
    ) -> tuple[int, int | None, int | None]:
        lease = await self.acquire(
            data_file=data_file,
        )
        try:
            meta = await lease.decode_window(
                from_timestamp_s=from_timestamp_s,
                to_timestamp_s=to_timestamp_s,
                on_frame=on_frame,
                decoder_threads=decoder_threads,
            )
        finally:
            lease.release()

        if meta.frame_count <= 0:
            raise ValueError(
                "Video segment contains no decodable frames in requested timestamp window."
            )
        return meta.frame_count, meta.width, meta.height

    async def resolve_fps(self, *, data_file: DataFile) -> int | None:
        lease = await self.acquire(data_file=data_file)
        try:
            return await lease.probe_fps()
        finally:
            lease.release()

    @staticmethod
    def _validate_bounds(
        *,
        from_timestamp_s: float,
        to_timestamp_s: float | None,
    ) -> tuple[float, float | None]:
        if not isinstance(from_timestamp_s, (int, float)) or from_timestamp_s < 0:
            raise ValueError("from_timestamp_s must be a non-negative number")

        start_ts = max(0.0, float(from_timestamp_s))
        end_ts = float(to_timestamp_s) if to_timestamp_s is not None else None
        if end_ts is not None and end_ts <= start_ts:
            raise ValueError(
                "Invalid video timestamp bounds: "
                f"from_timestamp_s={start_ts}, to_timestamp_s={end_ts}"
            )
        return start_ts, end_ts

    def _decode_with_decoder(
        self,
        *,
        container: Any,
        stream_index: int,
        from_timestamp_s: float,
        to_timestamp_s: float | None,
        on_frame: Callable[["av.VideoFrame"], None],
    ) -> DecodeWindowMeta:
        input_stream = container.streams.video[stream_index]
        stream_time_base = input_stream.time_base

        if stream_time_base is not None:
            seek_ts = int(from_timestamp_s / float(stream_time_base))
            try:
                container.seek(seek_ts, stream=input_stream)
            except Exception:
                try:
                    container.seek(max(0, seek_ts), any_frame=True, stream=input_stream)
                except Exception:
                    pass

        selected = 0
        width: int | None = None
        height: int | None = None

        epsilon = 1e-6
        for frame in container.decode(input_stream):
            ts = None
            if frame.pts is not None and frame.time_base is not None:
                ts = float(frame.pts * frame.time_base)
            if ts is None:
                continue
            if ts + epsilon < from_timestamp_s:
                continue
            if to_timestamp_s is not None and ts - epsilon >= to_timestamp_s:
                break

            if width is None:
                width = frame.width
                height = frame.height
            on_frame(frame)
            selected += 1

        return DecodeWindowMeta(
            frame_count=selected,
            width=width,
            height=height,
            pix_fmt="rgb24",
        )

    @staticmethod
    def _configure_decoder_threads(
        *,
        resource: _DecoderResource,
        decoder_threads: int | None,
    ) -> None:
        if decoder_threads is None:
            return
        requested_threads = int(decoder_threads)
        if resource.configured_decoder_threads == requested_threads:
            return
        if resource.configured_decoder_threads is not None:
            raise ValueError(
                "Decoder cache resource was already configured with "
                f"{resource.configured_decoder_threads} threads; use a distinct "
                "cache name for a different decoder thread policy."
            )
        codec_context = resource.container.streams.video[resource.stream_index].codec_context
        codec_context.thread_count = requested_threads
        codec_context.thread_type = "AUTO"
        resource.configured_decoder_threads = requested_threads

    def clear(self) -> None:
        self._cache.clear()


_decoder_cache_registry: dict[str, VideoDecoderCache] = {}


def get_video_decoder_cache(
    *,
    name: str = "default",
    media_cache: MediaLocalCache,
    max_decoders: int | None = None,
) -> VideoDecoderCache:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("decoder cache name must be a non-empty string")

    normalized = name.strip()
    cache = _decoder_cache_registry.get(normalized)
    if cache is None:
        cache = VideoDecoderCache(
            name=normalized,
            max_decoders=max_decoders if max_decoders is not None else 16,
            media_cache=media_cache,
        )
        _decoder_cache_registry[normalized] = cache
        return cache

    if max_decoders is not None and cache.max_decoders != int(max_decoders):
        raise ValueError(
            f"Decoder cache {normalized!r} already exists with "
            f"max_decoders={cache.max_decoders}"
        )
    return cache


def reset_video_decoder_cache(name: str | None = None) -> None:
    if name is None:
        items = list(_decoder_cache_registry.values())
        _decoder_cache_registry.clear()
    else:
        normalized = name.strip()
        cache = _decoder_cache_registry.pop(normalized, None)
        items = [cache] if cache is not None else []

    for cache in items:
        if cache is None:
            continue
        cache.clear()


__all__ = [
    "DecoderLease",
    "DecodeWindowMeta",
    "VideoDecoderCache",
    "get_video_decoder_cache",
    "reset_video_decoder_cache",
]
