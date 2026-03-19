from __future__ import annotations

import asyncio
import os
from refiner.io import DataFile
from refiner.media import VideoFile
from refiner.pipeline.utils.cache.file_cache import get_media_cache
from refiner.pipeline.utils.cache.lease_cache import LeaseCache


def test_media_cache_async_context_reuses_download() -> None:
    uri = "memory://context-manager.bin"
    with VideoFile(DataFile.resolve(uri)).open("wb") as f:
        f.write(b"context-manager")

    cache = get_media_cache("context-manager")

    async def exercise() -> tuple[str, str]:
        data_file = DataFile.resolve(uri)
        async with cache.cached(file=data_file) as first_path:
            assert os.path.exists(first_path)
            async with cache.cached(file=data_file) as second_path:
                assert os.path.exists(second_path)
                return first_path, second_path

    path, second_path = asyncio.run(exercise())
    assert path == second_path


def test_media_cache_file_lease_can_be_acquired_and_released_twice() -> None:
    uri = "memory://lease-same-file.bin"
    with VideoFile(DataFile.resolve(uri)).open("wb") as f:
        f.write(b"lease-reuse")

    cache = get_media_cache("lease-same-file")

    async def exercise() -> None:
        file = DataFile.resolve(uri)
        lease1 = await cache.acquire_file_lease(file)
        lease2 = await cache.acquire_file_lease(file)
        assert lease1.path == lease2.path
        lease1.release()
        lease2.release()

    asyncio.run(exercise())


def test_lease_cache_waiters_share_one_inflight_load() -> None:
    class _TestLeaseCache(LeaseCache[str, str]):
        def __init__(self) -> None:
            super().__init__(max_entries=4)
            self.create_calls = 0
            self.started = asyncio.Event()
            self.unblock = asyncio.Event()

        async def _create_resource(self, key: str) -> tuple[str, int]:
            self.create_calls += 1
            self.started.set()
            await self.unblock.wait()
            return f"resource:{key}", 1

        def _close_resource(self, resource: str) -> None:
            return

    async def exercise() -> None:
        cache = _TestLeaseCache()

        first = asyncio.create_task(cache.acquire("same"))
        await cache.started.wait()
        second = asyncio.create_task(cache.acquire("same"))

        cache.unblock.set()
        lease1, lease2 = await asyncio.gather(first, second)
        try:
            assert lease1.resource == "resource:same"
            assert lease2.resource == "resource:same"
            assert cache.create_calls == 1
        finally:
            lease1.release()
            lease2.release()

    asyncio.run(exercise())


def test_lease_cache_blocks_when_max_leases_per_key_is_reached() -> None:
    class _TestLeaseCache(LeaseCache[str, str]):
        def __init__(self) -> None:
            super().__init__(max_entries=4, max_leases_per_key=1)
            self.create_calls = 0

        async def _create_resource(self, key: str) -> tuple[str, int]:
            self.create_calls += 1
            return f"resource:{key}", 1

        def _close_resource(self, resource: str) -> None:
            return

    async def exercise() -> None:
        cache = _TestLeaseCache()
        lease1 = await cache.acquire("same")
        waiter = asyncio.create_task(cache.acquire("same"))
        await asyncio.sleep(0)
        assert not waiter.done()

        lease1.release()
        lease2 = await waiter
        try:
            assert lease2.resource == "resource:same"
            assert cache.create_calls == 1
        finally:
            lease2.release()

    asyncio.run(exercise())


def test_lease_cache_blocks_new_key_when_active_key_capacity_is_full() -> None:
    class _TestLeaseCache(LeaseCache[str, str]):
        def __init__(self) -> None:
            super().__init__(max_entries=1, block_on_capacity=True)
            self.create_calls = 0

        async def _create_resource(self, key: str) -> tuple[str, int]:
            self.create_calls += 1
            return f"resource:{key}", 1

        def _close_resource(self, resource: str) -> None:
            return

    async def exercise() -> None:
        cache = _TestLeaseCache()
        lease1 = await cache.acquire("one")
        waiter = asyncio.create_task(cache.acquire("two"))
        await asyncio.sleep(0)
        assert not waiter.done()

        lease1.release()
        lease2 = await waiter
        try:
            assert lease2.resource == "resource:two"
            assert cache.create_calls == 2
        finally:
            lease2.release()

    asyncio.run(exercise())


def test_lease_cache_blocks_new_key_when_weight_capacity_is_full() -> None:
    class _TestLeaseCache(LeaseCache[str, str]):
        def __init__(self) -> None:
            super().__init__(max_entries=4, max_weight=1, block_on_capacity=True)
            self.create_calls = 0

        async def _create_resource(self, key: str) -> tuple[str, int]:
            self.create_calls += 1
            return f"resource:{key}", 1

        def _close_resource(self, resource: str) -> None:
            return

    async def exercise() -> None:
        cache = _TestLeaseCache()
        lease1 = await cache.acquire("one")
        waiter = asyncio.create_task(cache.acquire("two"))
        await asyncio.sleep(0)
        assert not waiter.done()

        lease1.release()
        lease2 = await waiter
        try:
            assert lease2.resource == "resource:two"
            assert cache.create_calls == 2
        finally:
            lease2.release()

    asyncio.run(exercise())
