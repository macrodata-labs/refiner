from __future__ import annotations

import warnings

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]


def set_memory_soft_limit_mb(mem_mb: int) -> tuple[int, int] | None:
    if resource is None:
        warnings.warn(
            "mem_mb_per_worker requested but resource limits are unavailable; running without memory limit",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    target_bytes = int(mem_mb) * 1024 * 1024
    if target_bytes <= 0:
        raise ValueError("mem_mb_per_worker must be > 0")

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        target_bytes = min(target_bytes, hard)

    resource.setrlimit(resource.RLIMIT_AS, (target_bytes, hard))
    return (soft, hard)


def restore_memory_soft_limit(previous: tuple[int, int]) -> None:
    if resource is None:
        return
    soft, hard = previous
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


__all__ = ["set_memory_soft_limit_mb", "restore_memory_soft_limit"]
