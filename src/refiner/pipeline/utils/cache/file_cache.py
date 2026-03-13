from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from refiner.io import DataFile
from refiner.pipeline.utils.cache.lease_cache import CacheLease, LeaseCache


@dataclass(frozen=True, slots=True)
class _FileCacheKey:
    resolved: str
    file: DataFile

    def __hash__(self) -> int:
        return hash(self.resolved)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _FileCacheKey):
            return False
        return self.resolved == other.resolved


@dataclass(slots=True)
class _FileResource:
    path: str
    size_bytes: int


class _FileLeaseCache(LeaseCache[_FileCacheKey, _FileResource]):
    def __init__(
        self,
        *,
        name: str,
        max_entries: int,
        max_bytes: int,
    ) -> None:
        super().__init__(max_entries=max_entries, max_weight=max_bytes)
        self.name = name

    async def _create_resource(self, key: _FileCacheKey) -> tuple[_FileResource, int]:
        downloaded_path, downloaded_size_bytes = await asyncio.to_thread(
            _download_data_file_to_temp,
            key.file,
            cache_name=self.name,
        )
        resource = _FileResource(
            path=downloaded_path,
            size_bytes=max(0, int(downloaded_size_bytes)),
        )
        return resource, resource.size_bytes

    def _close_resource(self, resource: _FileResource) -> None:
        _safe_delete(resource.path)

    def _resource_is_valid(self, resource: _FileResource | None) -> bool:
        if resource is None:
            return False
        return Path(resource.path).exists()

    def evict_resolved(self, resolved_key: str) -> None:
        key = next((k for k in self._entries if k.resolved == resolved_key), None)
        if key is None:
            return
        entry = self._entries.get(key)
        if entry is None or entry.status != "ready" or entry.ref_count > 0:
            return
        self._drop_entry_unlocked(key)


@dataclass(slots=True)
class _CacheFileLease:
    _lease: CacheLease[_FileCacheKey, _FileResource]
    path: str
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._lease.release()


class _CachedFileContext:
    def __init__(self, cache: "MediaLocalCache", file: DataFile) -> None:
        self._cache = cache
        self._file = file
        self._lease: _CacheFileLease | None = None

    async def __aenter__(self) -> str:
        self._lease = await self._cache.acquire_file_lease(self._file)
        return self._lease.path

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        if self._lease is not None:
            self._lease.release()
            self._lease = None

    def __enter__(self) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._lease = asyncio.run(self._cache.acquire_file_lease(self._file))
            return self._lease.path

        raise RuntimeError(
            "Cannot use sync cache context manager while an event loop is running. "
            "Use 'async with cache.cached(...)' instead."
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        if self._lease is not None:
            self._lease.release()
            self._lease = None


class MediaLocalCache:
    """Named local file cache with in-flight download dedupe and LRU eviction."""

    def __init__(
        self,
        *,
        name: str,
        max_entries: int = 128,
        max_bytes: int = 1_024 * 1_024 * 1_024,
        max_inflight_downloads: int | None = None,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        resolved_inflight = (
            max_entries
            if max_inflight_downloads is None
            else int(max_inflight_downloads)
        )
        if resolved_inflight <= 0:
            raise ValueError("max_inflight_downloads must be > 0")
        self.name = name
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.max_inflight_downloads = resolved_inflight
        self._lease_slots = asyncio.BoundedSemaphore(self.max_inflight_downloads)
        self._cache = _FileLeaseCache(
            name=name,
            max_entries=self.max_entries,
            max_bytes=self.max_bytes,
        )

    def cached(
        self,
        *,
        file: DataFile,
    ) -> _CachedFileContext:
        return _CachedFileContext(cache=self, file=file)

    async def get_lease(self) -> None:
        await self._lease_slots.acquire()

    def release_lease(self) -> None:
        self._lease_slots.release()

    async def acquire_file_lease(self, file: DataFile) -> _CacheFileLease:
        key = _FileCacheKey(resolved=str(file), file=file)
        lease = await self._cache.acquire(key)
        return _CacheFileLease(lease, path=lease.resource.path)

    def evict(self, key: str) -> None:
        self._cache.evict_resolved(key)

    def clear(self) -> None:
        self._cache.clear()


_CACHE_REGISTRY: dict[str, MediaLocalCache] = {}


def get_media_cache(
    name: str = "default",
    *,
    max_entries: int | None = None,
    max_bytes: int | None = None,
    max_inflight_downloads: int | None = None,
) -> MediaLocalCache:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("cache name must be a non-empty string")

    normalized = name.strip()
    cache = _CACHE_REGISTRY.get(normalized)
    if cache is None:
        cache = MediaLocalCache(
            name=normalized,
            max_entries=max_entries if max_entries is not None else 128,
            max_bytes=max_bytes if max_bytes is not None else 1_024 * 1_024 * 1_024,
            max_inflight_downloads=max_inflight_downloads,
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
    if (
        max_inflight_downloads is not None
        and cache.max_inflight_downloads != int(max_inflight_downloads)
    ):
        raise ValueError(
            f"Cache {normalized!r} already exists with "
            f"max_inflight_downloads={cache.max_inflight_downloads}"
        )
    return cache


def reset_media_cache(name: str | None = None) -> None:
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


# TODO: Make this async
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
    "FileCache",
    "MediaLocalCache",
    "get_media_cache",
    "reset_media_cache",
]


FileCache = MediaLocalCache
