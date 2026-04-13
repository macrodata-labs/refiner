from __future__ import annotations

import os


def parse_cpu_ids(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(x) for x in raw.split(",") if x.strip()]


def available_cpu_ids() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(x) for x in os.sched_getaffinity(0))
    count = os.cpu_count()
    if count is None or count <= 0:
        raise RuntimeError("Unable to determine available CPUs")
    return list(range(int(count)))


__all__ = [
    "available_cpu_ids",
    "parse_cpu_ids",
]
