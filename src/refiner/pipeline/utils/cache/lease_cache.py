from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, Hashable, Literal, TypeVar


K = TypeVar("K", bound=Hashable)
R = TypeVar("R")

CacheEntryStatus = Literal["loading", "ready", "error"]


@dataclass(slots=True)
class _LeaseCacheEntry(Generic[R]):
    status: CacheEntryStatus
    resource: R | None
    error: BaseException | None = None
    ref_count: int = 0
    weight: int = 0


class CacheLease(Generic[K, R]):
    def __init__(
        self,
        *,
        cache: "LeaseCache[K, R]",
        key: K,
        resource: R,
    ) -> None:
        self._cache = cache
        self._key = key
        self.resource = resource
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._cache._release(self._key)


class LeaseCache(Generic[K, R]):
    """Shared async lease+eviction engine for keyed resources."""

    def __init__(
        self,
        *,
        max_entries: int,
        max_weight: int | None = None,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        if max_weight is not None and max_weight <= 0:
            raise ValueError("max_weight must be > 0 when provided")
        self.max_entries = int(max_entries)
        self.max_weight = int(max_weight) if max_weight is not None else None
        self._entries: OrderedDict[K, _LeaseCacheEntry[R]] = OrderedDict()
        self._total_weight = 0

    async def acquire(self, key: K) -> CacheLease[K, R]:
        while True:
            entry = self._entries.get(key)
            if entry is None:
                self._entries[key] = _LeaseCacheEntry(
                    status="loading",
                    resource=None,
                )
                self._entries.move_to_end(key)
                should_create = True
            elif entry.status == "ready":
                if not self._resource_is_valid(entry.resource):
                    self._drop_entry_unlocked(key)
                    continue
                entry.ref_count += 1
                self._entries.move_to_end(key)
                return CacheLease(cache=self, key=key, resource=entry.resource)  # type: ignore[arg-type]
            elif entry.status == "error":
                err = entry.error
                self._entries.pop(key, None)
                if err is None:
                    raise RuntimeError("cache entry failed without an error")
                raise err
            else:
                should_create = False

            if should_create:
                try:
                    resource, weight = await self._create_resource(key)
                except BaseException as exc:
                    self._set_entry_error(key=key, exc=exc)
                    raise

                current = self._entries.get(key)
                if current is not None and current.status == "loading":
                    current.resource = resource
                    current.weight = max(0, int(weight))
                    current.status = "ready"
                    current.error = None
                    current.ref_count += 1
                    self._entries.move_to_end(key)
                    self._total_weight += current.weight
                    self._evict_unlocked()
                    return CacheLease(cache=self, key=key, resource=resource)

                self._close_resource(resource)
                continue

            await asyncio.sleep(0.001)

    def evict(self, key: K) -> None:
        entry = self._entries.get(key)
        if entry is None or entry.status != "ready" or entry.ref_count > 0:
            return
        self._drop_entry_unlocked(key)

    def clear(self) -> None:
        entries = list(self._entries.values())
        self._entries.clear()
        self._total_weight = 0

        for entry in entries:
            if entry.status == "loading":
                entry.error = RuntimeError("cache was cleared while resource load was in-flight")
                continue
            if entry.status == "ready" and entry.resource is not None:
                self._close_resource(entry.resource)

    def _release(self, key: K) -> None:
        entry = self._entries.get(key)
        if entry is None or entry.status != "ready":
            return
        entry.ref_count = max(0, entry.ref_count - 1)
        self._evict_unlocked()

    def _set_entry_error(self, *, key: K, exc: BaseException) -> None:
        entry = self._entries.get(key)
        if entry is None or entry.status != "loading":
            return
        entry.status = "error"
        entry.error = exc

    def _evict_unlocked(self) -> None:
        while self._over_limit_unlocked():
            candidates = [
                k
                for k, entry in self._entries.items()
                if entry.status == "ready" and entry.ref_count == 0
            ]
            if not candidates:
                return
            self._drop_entry_unlocked(candidates[0])

    def _drop_entry_unlocked(self, key: K) -> None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        self._total_weight = max(0, self._total_weight - int(entry.weight))
        if entry.status == "ready" and entry.resource is not None:
            self._close_resource(entry.resource)

    def _over_limit_unlocked(self) -> bool:
        ready_count = sum(1 for entry in self._entries.values() if entry.status == "ready")
        if ready_count > self.max_entries:
            return True
        if self.max_weight is not None and self._total_weight > self.max_weight:
            return True
        return False

    async def _create_resource(self, key: K) -> tuple[R, int]:
        raise NotImplementedError

    def _close_resource(self, resource: R) -> None:
        raise NotImplementedError

    def _resource_is_valid(self, resource: R | None) -> bool:
        return resource is not None
