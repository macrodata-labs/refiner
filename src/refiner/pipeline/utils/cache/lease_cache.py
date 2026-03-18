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
    ready_future: asyncio.Future[None]
    available_future: asyncio.Future[None] | None = None
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
        max_leases_per_key: int | None = None,
        block_on_capacity: bool = False,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        if max_weight is not None and max_weight <= 0:
            raise ValueError("max_weight must be > 0 when provided")
        if max_leases_per_key is not None and max_leases_per_key <= 0:
            raise ValueError("max_leases_per_key must be > 0 when provided")
        self.max_entries = int(max_entries)
        self.max_weight = int(max_weight) if max_weight is not None else None
        self.max_leases_per_key = (
            int(max_leases_per_key) if max_leases_per_key is not None else None
        )
        self.block_on_capacity = bool(block_on_capacity)
        self._entries: OrderedDict[K, _LeaseCacheEntry[R]] = OrderedDict()
        self._total_weight = 0
        self._capacity_future: asyncio.Future[None] | None = None

    async def acquire(self, key: K) -> CacheLease[K, R]:
        while True:
            wait_future: asyncio.Future[None] | None = None
            entry = self._entries.get(key)
            should_create = False
            if entry is None:
                if self.block_on_capacity and self._at_capacity_for_new_key_unlocked():
                    if self._drop_first_idle_unlocked():
                        continue
                    wait_future = self._capacity_wait_future_unlocked()
                else:
                    self._entries[key] = _LeaseCacheEntry(
                        status="loading",
                        resource=None,
                        ready_future=asyncio.get_running_loop().create_future(),
                    )
                    self._entries.move_to_end(key)
                    should_create = True
            elif entry.status == "ready":
                if not self._resource_is_valid(entry.resource):
                    self._drop_entry_unlocked(key)
                    continue
                if self._can_issue_lease_unlocked(entry):
                    entry.ref_count += 1
                    self._entries.move_to_end(key)
                    return CacheLease(cache=self, key=key, resource=entry.resource)  # type: ignore[arg-type]
                wait_future = self._entry_available_future_unlocked(entry)
            elif entry.status == "error":
                err = entry.error
                self._entries.pop(key, None)
                if err is None:
                    raise RuntimeError("cache entry failed without an error")
                raise err
            else:
                wait_future = entry.ready_future

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
                    if not current.ready_future.done():
                        current.ready_future.set_result(None)
                    self._entries.move_to_end(key)
                    self._total_weight += current.weight
                    self._trim_idle_to_capacity_unlocked()
                    return CacheLease(cache=self, key=key, resource=resource)

                self._close_resource(resource)
                continue

            if wait_future is None:
                raise RuntimeError("loading cache entry missing ready future")
            await wait_future

    def evict(self, key: K) -> None:
        entry = self._entries.get(key)
        if entry is None or entry.status != "ready" or entry.ref_count > 0:
            return
        self._drop_entry_unlocked(key)

    def clear(self) -> None:
        entries = list(self._entries.values())
        self._entries.clear()
        self._total_weight = 0
        capacity_future = self._capacity_future
        self._capacity_future = None

        for entry in entries:
            if entry.available_future is not None and not entry.available_future.done():
                entry.available_future.set_exception(
                    RuntimeError("cache was cleared while waiters were blocked")
                )
            if entry.status == "loading":
                error = RuntimeError(
                    "cache was cleared while resource load was in-flight"
                )
                entry.error = error
                if not entry.ready_future.done():
                    entry.ready_future.set_exception(error)
                continue
            if entry.status == "ready" and entry.resource is not None:
                self._close_resource(entry.resource)
        if capacity_future is not None and not capacity_future.done():
            capacity_future.set_exception(
                RuntimeError("cache was cleared while waiting for capacity")
            )

    def _release(self, key: K) -> None:
        entry = self._entries.get(key)
        if entry is None or entry.status != "ready":
            return
        entry.ref_count = max(0, entry.ref_count - 1)
        self._notify_entry_available_unlocked(entry)
        self._notify_capacity_available_unlocked()

    def _set_entry_error(self, *, key: K, exc: BaseException) -> None:
        entry = self._entries.get(key)
        if entry is None or entry.status != "loading":
            return
        entry.status = "error"
        entry.error = exc
        if not entry.ready_future.done():
            entry.ready_future.set_exception(exc)

    def _drop_entry_unlocked(self, key: K) -> None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        self._total_weight = max(0, self._total_weight - int(entry.weight))
        self._notify_entry_available_unlocked(entry)
        if entry.status == "ready" and entry.resource is not None:
            self._close_resource(entry.resource)
        self._notify_capacity_available_unlocked()

    def _over_limit_unlocked(self) -> bool:
        ready_count = sum(
            1 for entry in self._entries.values() if entry.status == "ready"
        )
        if ready_count > self.max_entries:
            return True
        if self.max_weight is not None and self._total_weight > self.max_weight:
            return True
        return False

    def _at_capacity_for_new_key_unlocked(self) -> bool:
        if len(self._entries) >= self.max_entries:
            return True
        if self.max_weight is not None and self._total_weight >= self.max_weight:
            return True
        return False

    def _drop_first_idle_unlocked(self) -> bool:
        candidate = next(
            (
                key
                for key, entry in self._entries.items()
                if entry.status == "ready"
                and entry.ref_count == 0
                and entry.available_future is None
            ),
            None,
        )
        if candidate is None:
            return False
        self._drop_entry_unlocked(candidate)
        return True

    def _trim_idle_to_capacity_unlocked(self) -> None:
        while self._over_limit_unlocked():
            if not self._drop_first_idle_unlocked():
                return

    def _can_issue_lease_unlocked(self, entry: _LeaseCacheEntry[R]) -> bool:
        if self.max_leases_per_key is None:
            return True
        return entry.ref_count < self.max_leases_per_key

    def _entry_available_future_unlocked(
        self,
        entry: _LeaseCacheEntry[R],
    ) -> asyncio.Future[None]:
        future = entry.available_future
        if future is None or future.done():
            future = asyncio.get_running_loop().create_future()
            entry.available_future = future
        return future

    def _notify_entry_available_unlocked(self, entry: _LeaseCacheEntry[R]) -> None:
        future = entry.available_future
        entry.available_future = None
        if future is not None and not future.done():
            future.set_result(None)

    def _capacity_wait_future_unlocked(self) -> asyncio.Future[None]:
        future = self._capacity_future
        if future is None or future.done():
            future = asyncio.get_running_loop().create_future()
            self._capacity_future = future
        return future

    def _notify_capacity_available_unlocked(self) -> None:
        future = self._capacity_future
        self._capacity_future = None
        if future is not None and not future.done():
            future.set_result(None)

    async def _create_resource(self, key: K) -> tuple[R, int]:
        raise NotImplementedError

    def _close_resource(self, resource: R) -> None:
        raise NotImplementedError

    def _resource_is_valid(self, resource: R | None) -> bool:
        return resource is not None
