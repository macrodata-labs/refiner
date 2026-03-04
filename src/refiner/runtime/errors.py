from __future__ import annotations


class UserMetricsFlushError(RuntimeError):
    """Raised when completed-shard user metrics flush fails."""


__all__ = ["UserMetricsFlushError"]
