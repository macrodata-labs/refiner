from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


CacheEntryStatus = Literal["loading", "ready", "error"]


@dataclass(slots=True)
class _DecoderEntry:
    status: CacheEntryStatus
    container: Any | None
    stream_index: int | None
    lock: threading.Lock
    event: threading.Event
    error: BaseException | None = None


class VideoDecoderCache:
    """Named in-memory cache for decoded AV containers.

    The cache keeps AV containers open and serializes access to each container with a
    lock to allow safe reuse from worker threads while preserving seek+decode state.
    """

    def __init__(
        self,
        *,
        name: str,
        max_decoders: int = 16,
    ) -> None:
        if max_decoders <= 0:
            raise ValueError("max_decoders must be > 0")

        import av

        self.name = name
        self.max_decoders = int(max_decoders)
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _DecoderEntry] = OrderedDict()
        self._av = av

    def decode_segment(
        self,
        *,
        local_path: str,
        from_timestamp_s: float,
        to_timestamp_s: float | None,
    ) -> tuple[tuple[np.ndarray, ...], int | None, int | None, str | None]:
        if not local_path:
            raise ValueError("local_path cannot be empty")
        if not isinstance(from_timestamp_s, (int, float)) or from_timestamp_s < 0:
            raise ValueError("from_timestamp_s must be a non-negative number")

        start_ts = max(0.0, float(from_timestamp_s))
        end_ts = float(to_timestamp_s) if to_timestamp_s is not None else None
        if end_ts is not None and end_ts <= start_ts:
            raise ValueError(
                "Invalid video timestamp bounds: "
                f"from_timestamp_s={start_ts}, to_timestamp_s={end_ts}"
            )

        while True:
            with self._lock:
                entry = self._entries.get(local_path)
                if entry is None:
                    entry = _DecoderEntry(
                        status="loading",
                        container=None,
                        stream_index=None,
                        lock=threading.Lock(),
                        event=threading.Event(),
                    )
                    self._entries[local_path] = entry
                    self._entries.move_to_end(local_path)
                    should_create = True
                elif entry.status == "ready":
                    self._entries.move_to_end(local_path)
                    break
                elif entry.status == "error":
                    err = entry.error
                    self._entries.pop(local_path, None)
                    if err is None:
                        raise RuntimeError("decoder entry failed without an error")
                    raise err
                else:
                    should_create = False
                    loader = entry.event

            if should_create:
                with self._lock:
                    # Another thread may have initialized while we waited for lock.
                    existing = self._entries.get(local_path)
                    if existing is not None and existing.status == "ready":
                        entry = existing
                        break

                try:
                    container = self._av.open(local_path)
                    stream = container.streams.video[0]
                except Exception as exc:
                    with self._lock:
                        existing = self._entries.get(local_path)
                        if existing is not None and existing.status == "loading":
                            existing.status = "error"
                            existing.error = exc
                            existing.event.set()
                    raise

                with self._lock:
                    existing = self._entries.get(local_path)
                    if existing is not None and existing.status == "loading":
                        existing.container = container
                        existing.stream_index = int(stream.index)
                        existing.status = "ready"
                        existing.event.set()
                    else:
                        # Another thread won initialization first.
                        container.close()
                    self._entries.move_to_end(local_path)
                    self._evict_unlocked()

                if existing is not None and existing.status == "ready":
                    entry = existing
                    break
            else:
                loader.wait()
                continue

        if not isinstance(entry, _DecoderEntry) or entry.status != "ready":
            raise RuntimeError("Decoder entry was not prepared")
        if entry.container is None or entry.stream_index is None:
            raise RuntimeError("Decoder entry is missing container state")

        with entry.lock:
            frames = self._decode_with_decoder(
                container=entry.container,
                stream_index=entry.stream_index,
                from_timestamp_s=start_ts,
                to_timestamp_s=end_ts,
            )
        return frames

    def _decode_with_decoder(
        self,
        *,
        container: Any,
        stream_index: int,
        from_timestamp_s: float,
        to_timestamp_s: float | None,
    ) -> tuple[tuple[np.ndarray, ...], int | None, int | None, str | None]:
        input_stream = container.streams.video[stream_index]
        stream_time_base = input_stream.time_base

        if stream_time_base is not None:
            seek_ts = int(from_timestamp_s / float(stream_time_base))
            try:
                container.seek(seek_ts, stream=input_stream)
            except Exception:
                # Fallback to best-effort seek at lower precision.
                try:
                    container.seek(max(0, seek_ts), any_frame=True, stream=input_stream)
                except Exception:
                    pass

        out_frames: list[np.ndarray] = []
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

            frame_rgb = frame.to_ndarray(format="rgb24")
            if width is None:
                width = frame.width
                height = frame.height
            out_frames.append(frame_rgb)

        if not out_frames:
            raise ValueError(
                "Video segment contains no decodable frames in requested timestamp window."
            )

        return tuple(out_frames), width, height, "rgb24"

    def clear(self) -> None:
        with self._lock:
            for entry in self._entries.values():
                if entry.status == "ready" and entry.container is not None:
                    entry.container.close()
            self._entries.clear()

    def _evict_unlocked(self) -> None:
        while len(self._entries) > self.max_decoders:
            keys = [
                key for key, entry in self._entries.items() if entry.status == "ready"
            ]
            if not keys:
                return
            for key in keys[:1]:
                self._drop_entry_unlocked(key)
                break

    def _drop_entry_unlocked(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        if entry.status == "ready" and entry.container is not None:
            entry.container.close()


_decoder_cache_registry: dict[str, VideoDecoderCache] = {}
_decoder_cache_registry_lock = threading.Lock()


def get_video_decoder_cache(
    *,
    name: str = "default",
    max_decoders: int | None = None,
) -> VideoDecoderCache:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("decoder cache name must be a non-empty string")

    normalized = name.strip()
    with _decoder_cache_registry_lock:
        cache = _decoder_cache_registry.get(normalized)
        if cache is None:
            cache = VideoDecoderCache(
                name=normalized,
                max_decoders=max_decoders if max_decoders is not None else 16,
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
    with _decoder_cache_registry_lock:
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
    "VideoDecoderCache",
    "get_video_decoder_cache",
    "reset_video_decoder_cache",
]
