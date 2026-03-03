from __future__ import annotations

import os
import warnings


def available_cpu_ids() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(x) for x in os.sched_getaffinity(0))
    count = os.cpu_count()
    if count is None or count <= 0:
        raise RuntimeError("Unable to determine available CPUs")
    return list(range(int(count)))


def build_cpu_sets(*, num_workers: int, cpus_per_worker: int) -> list[list[int]]:
    if cpus_per_worker <= 0:
        raise ValueError("cpus_per_worker must be > 0")
    cpu_ids = available_cpu_ids()
    needed: int = num_workers * cpus_per_worker
    if needed > len(cpu_ids):
        raise ValueError(
            f"Requested {needed} CPUs ({num_workers} workers x {cpus_per_worker}) but only {len(cpu_ids)} are available"
        )
    out: list[list[int]] = []
    for i in range(num_workers):
        start = i * cpus_per_worker
        out.append(cpu_ids[start : start + cpus_per_worker])
    return out


def set_cpu_affinity(cpu_ids: list[int]) -> None:
    if not cpu_ids:
        return
    if not hasattr(os, "sched_setaffinity"):
        warnings.warn(
            "cpus_per_worker requested but os.sched_setaffinity is not available; running without CPU pinning",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    try:
        os.sched_setaffinity(0, set(cpu_ids))
    except Exception as e:
        warnings.warn(
            f"Failed to set CPU affinity ({e}); running without CPU pinning",
            RuntimeWarning,
            stacklevel=2,
        )


__all__ = ["available_cpu_ids", "build_cpu_sets", "set_cpu_affinity"]
