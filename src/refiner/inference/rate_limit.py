from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StaticRateLimit:
    """Fixed request concurrency settings for inference providers."""

    max_concurrency: int = 256

    def __post_init__(self) -> None:
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")


@dataclass(frozen=True, slots=True)
class AdaptiveRateLimit:
    """Adaptive request concurrency settings for inference providers."""

    max_concurrency: int = 256
    min_concurrency: int = 1
    initial_concurrency: int | None = None
    initial_growth_multiplier: float = 2.0
    min_growth_multiplier: float = 1.05
    growth_multiplier_step: float = 0.1
    rate_limit_decrease_factor: float = 0.5
    success_window_requests: int = 50
    default_cooldown_seconds: float = 1.0

    def __post_init__(self) -> None:
        if (
            self.max_concurrency <= 0
            or self.min_concurrency <= 0
            or self.success_window_requests <= 0
        ):
            raise ValueError("concurrency and success window must be > 0")
        if self.max_concurrency < self.min_concurrency:
            raise ValueError("max_concurrency must be >= min_concurrency")
        if self.initial_concurrency is not None:
            if self.initial_concurrency < self.min_concurrency:
                raise ValueError("initial_concurrency must be >= min_concurrency")
        if not 1 <= self.min_growth_multiplier <= self.initial_growth_multiplier:
            raise ValueError("growth multipliers must satisfy 1 <= min <= initial")
        if self.growth_multiplier_step < 0 or self.default_cooldown_seconds < 0:
            raise ValueError("growth step and cooldown must be >= 0")
        if not 0 < self.rate_limit_decrease_factor < 1:
            raise ValueError("rate_limit_decrease_factor must be > 0 and < 1")


class AdaptiveRateLimiter:
    """Async limiter that probes provider capacity and backs off on rate limits."""

    def __init__(self, config: AdaptiveRateLimit) -> None:
        self._config = config
        initial = config.initial_concurrency
        if initial is None:
            initial = min(16, config.max_concurrency)
        self._limit = min(max(initial, config.min_concurrency), config.max_concurrency)
        self._growth_multiplier = config.initial_growth_multiplier
        self._running = 0
        self._successes = 0
        self._cooldown_until = 0.0
        self._condition = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    async def acquire(self) -> None:
        async with self._condition:
            while True:
                cooldown_remaining = self._cooldown_until - time.monotonic()
                if cooldown_remaining > 0:
                    try:
                        await asyncio.wait_for(
                            self._condition.wait(),
                            timeout=cooldown_remaining,
                        )
                    except TimeoutError:
                        pass
                    continue
                if self._running < self._limit:
                    self._running += 1
                    return
                await self._condition.wait()

    async def release(self) -> None:
        async with self._condition:
            self._running -= 1
            self._condition.notify_all()

    async def record_success(self) -> None:
        async with self._condition:
            self._successes += 1
            if self._successes < self._config.success_window_requests:
                return
            self._successes = 0
            next_limit = min(
                self._config.max_concurrency,
                max(self._limit + 1, math.ceil(self._limit * self._growth_multiplier)),
            )
            if next_limit <= self._limit:
                return
            self._limit = next_limit
            self._growth_multiplier = self._reduced_multiplier(
                self._config.growth_multiplier_step
            )
            self._condition.notify_all()

    async def record_rate_limit(self, retry_after_seconds: float | None = None) -> None:
        async with self._condition:
            self._successes = 0
            self._limit = max(
                self._config.min_concurrency,
                math.floor(self._limit * self._config.rate_limit_decrease_factor),
            )
            self._growth_multiplier = self._reduced_multiplier(
                self._config.growth_multiplier_step * 2
            )
            cooldown = (
                retry_after_seconds
                if retry_after_seconds is not None
                else self._config.default_cooldown_seconds
            )
            if cooldown > 0:
                self._cooldown_until = max(
                    self._cooldown_until,
                    time.monotonic() + cooldown,
                )
            self._condition.notify_all()

    def _reduced_multiplier(self, step: float) -> float:
        return max(
            self._config.min_growth_multiplier,
            self._growth_multiplier - step,
        )


RateLimit = StaticRateLimit | AdaptiveRateLimit


__all__ = ["AdaptiveRateLimit", "AdaptiveRateLimiter", "RateLimit", "StaticRateLimit"]
