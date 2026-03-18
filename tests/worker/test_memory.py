from __future__ import annotations

import pytest

from refiner.worker.resources import memory


def test_set_memory_soft_limit_rejects_limits_below_current_vms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory, "_current_virtual_memory_bytes", lambda: 512 * 1024**2)
    monkeypatch.setattr(
        memory.resource,
        "getrlimit",
        lambda _: (memory.resource.RLIM_INFINITY, memory.resource.RLIM_INFINITY),
    )

    with pytest.raises(
        ValueError,
        match=r"mem_mb_per_worker=256 MB is below the worker's current virtual memory footprint",
    ):
        memory.set_memory_soft_limit_mb(256)


def test_set_memory_soft_limit_applies_when_above_current_vms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(memory, "_current_virtual_memory_bytes", lambda: 256 * 1024**2)
    monkeypatch.setattr(
        memory.resource,
        "getrlimit",
        lambda _: (123, memory.resource.RLIM_INFINITY),
    )
    monkeypatch.setattr(
        memory.resource,
        "setrlimit",
        lambda limit, value: calls.append((limit, value)),
    )

    previous = memory.set_memory_soft_limit_mb(512)

    assert previous == (123, memory.resource.RLIM_INFINITY)
    assert calls == [
        (
            memory.resource.RLIMIT_AS,
            (512 * 1024**2, memory.resource.RLIM_INFINITY),
        )
    ]
