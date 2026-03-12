from __future__ import annotations

import os
from contextlib import contextmanager
import shutil
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from refiner.io import DataFile


CacheEntryStatus = Literal["loading", "ready", "error"]


@dataclass(slots=True)
class _CacheEntry:
    status: CacheEntryStatus
    path: str | None
    size_bytes: int
    event: threading.Event
    error: BaseException | None = None
    ref_count: int = 0


class MediaLocalCache:
    """Named local file cache with in-flight download dedupe and LRU eviction."""

    def __init__(
        self,
        *,
        name: str,
        max_entries: int = 128,
        max_bytes: int = 1_024 * 1_024 * 1_024,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        self.name = name
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._total_bytes = 0

    @contextmanager
    def cached(
        self,
        *,
        file: DataFile,
    ) -> Iterator[str]:
        resolved_key = str(file)
        while True:
            wait_event: threading.Event | None = None
            should_download = False
            owned_path: str | None = None

            with self._lock:
                entry = self._entries.get(resolved_key)
                if entry is None:
                    entry = _CacheEntry(
                        status="loading",
                        path=None,
                        size_bytes=0,
                        event=threading.Event(),
                    )
                    self._entries[resolved_key] = entry
                    self._entries.move_to_end(resolved_key)
                    should_download = True
                elif entry.status == "loading":
                    wait_event = entry.event
                elif entry.status == "error":
                    err = entry.error
                    self._entries.pop(resolved_key, None)
                    if err is None:
                        raise RuntimeError("cache entry failed without an error")
                    raise err
                else:
                    if entry.path is None:
                        self._entries.pop(resolved_key, None)
                        continue
                    if not Path(entry.path).exists():
                        self._drop_entry_unlocked(
                            key=resolved_key,
                            entry=entry,
                            delete_file=False,
                        )
                        continue
                    entry.ref_count += 1
                    self._entries.move_to_end(resolved_key)
                    owned_path = entry.path

            if owned_path is not None:
                try:
                    yield owned_path
                finally:
                    with self._lock:
                        current = self._entries.get(resolved_key)
                        if current is not None and current.status == "ready":
                            current.ref_count = max(0, current.ref_count - 1)
                            self._evict_unlocked()
                return

            if wait_event is not None:
                wait_event.wait()
                continue

            if not should_download:
                continue

            try:
                downloaded_path, downloaded_size_bytes = _download_data_file_to_temp(
                    file,
                    cache_name=self.name,
                )
            except BaseException as exc:
                with self._lock:
                    current = self._entries.get(resolved_key)
                    if current is not None and current.status == "loading":
                        current.status = "error"
                        current.error = exc
                        current.event.set()
                raise

            with self._lock:
                current = self._entries.get(resolved_key)
                if current is None:
                    _safe_delete(downloaded_path)
                    continue

                if current.status == "loading":
                    current.status = "ready"
                    current.path = downloaded_path
                    current.size_bytes = max(0, int(downloaded_size_bytes))
                    current.error = None
                    current.event.set()
                    self._total_bytes += current.size_bytes
                    self._entries.move_to_end(resolved_key)
                    self._evict_unlocked()
                    continue

                _safe_delete(downloaded_path)
                continue

    def evict(self, key: str) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            if entry.status != "ready":
                return
            self._drop_entry_unlocked(key, entry)

    def clear(self) -> None:
        with self._lock:
            for entry in self._entries.values():
                if entry.status == "ready" and entry.path is not None:
                    _safe_delete(entry.path)
            self._entries.clear()
            self._total_bytes = 0

    def _evict_unlocked(self) -> None:
        while self._over_limit_unlocked():
            entries_to_evict = [
                (key, entry)
                for key, entry in list(self._entries.items())
                if entry.status == "ready" and entry.ref_count == 0
            ]
            if not entries_to_evict:
                return

            key, entry = entries_to_evict[0]
            self._drop_entry_unlocked(key, entry)

    def _over_limit_unlocked(self) -> bool:
        if len(self._entries) > self.max_entries:
            return True
        return self._total_bytes > self.max_bytes

    def _drop_entry_unlocked(
        self,
        key: str,
        entry: _CacheEntry,
        *,
        delete_file: bool = True,
    ) -> None:
        self._entries.pop(key, None)
        self._total_bytes = max(0, self._total_bytes - entry.size_bytes)
        if delete_file and entry.path is not None:
            _safe_delete(entry.path)


_CACHE_REGISTRY: dict[str, MediaLocalCache] = {}
_CACHE_REGISTRY_LOCK = threading.Lock()


def get_media_cache(
    name: str = "default",
    *,
    max_entries: int | None = None,
    max_bytes: int | None = None,
) -> MediaLocalCache:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("cache name must be a non-empty string")

    normalized = name.strip()
    with _CACHE_REGISTRY_LOCK:
        cache = _CACHE_REGISTRY.get(normalized)
        if cache is None:
            cache = MediaLocalCache(
                name=normalized,
                max_entries=max_entries if max_entries is not None else 128,
                max_bytes=max_bytes if max_bytes is not None else 1_024 * 1_024 * 1_024,
            )
            _CACHE_REGISTRY[normalized] = cache
            return cache

        if max_entries is not None and cache.max_entries != int(max_entries):
            raise ValueError(
                f"Cache {normalized!r} already exists with max_entries={cache.max_entries}"
            )
        if max_bytes is not None and cache.max_bytes != int(max_bytes):
            raise ValueError(
                f"Cache {normalized!r} already exists with max_bytes={cache.max_bytes}"
            )
        return cache


def reset_media_cache(name: str | None = None) -> None:
    with _CACHE_REGISTRY_LOCK:
        if name is None:
            items = list(_CACHE_REGISTRY.items())
            _CACHE_REGISTRY.clear()
        else:
            normalized = name.strip()
            cache = _CACHE_REGISTRY.pop(normalized, None)
            items = [(normalized, cache)] if cache is not None else []

    for _, cache in items:
        if cache is None:
            continue
        cache.clear()


def _safe_delete(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _download_data_file_to_temp(
    file: DataFile,
    *,
    cache_name: str,
) -> tuple[str, int]:
    file_suffix = os.path.splitext(file.path)[1] or ".bin"
    prefix = f"refiner_media_{cache_name}_"
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=file_suffix)
    os.close(fd)
    try:
        with file.open("rb") as src, open(temp_path, "wb") as dst:
            shutil.copyfileobj(fsrc=src, fdst=dst, length=8 * 1024 * 1024)
    except Exception:
        _safe_delete(temp_path)
        raise
    return temp_path, os.path.getsize(temp_path)


__all__ = [
    "MediaLocalCache",
    "get_media_cache",
    "reset_media_cache",
]
