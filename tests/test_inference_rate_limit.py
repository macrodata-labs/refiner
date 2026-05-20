from __future__ import annotations

import asyncio

from refiner.inference.rate_limit import AdaptiveRateLimit, AdaptiveRateLimiter


def test_adaptive_rate_limiter_grows_with_shrinking_multiplier() -> None:
    limiter = AdaptiveRateLimiter(
        AdaptiveRateLimit(
            max_concurrency=100,
            initial_concurrency=10,
            success_window_requests=1,
        )
    )

    asyncio.run(limiter.record_success())
    assert limiter.limit == 20

    asyncio.run(limiter.record_success())
    assert limiter.limit == 38


def test_adaptive_rate_limiter_halves_on_rate_limit() -> None:
    limiter = AdaptiveRateLimiter(
        AdaptiveRateLimit(
            max_concurrency=100,
            initial_concurrency=80,
            success_window_requests=1,
            default_cooldown_seconds=0,
        )
    )

    asyncio.run(limiter.record_rate_limit())

    assert limiter.limit == 40

    asyncio.run(limiter.record_success())
    assert limiter.limit == 72


def test_adaptive_rate_limiter_never_drops_below_minimum() -> None:
    limiter = AdaptiveRateLimiter(
        AdaptiveRateLimit(
            max_concurrency=100,
            min_concurrency=3,
            initial_concurrency=4,
            default_cooldown_seconds=0,
        )
    )

    asyncio.run(limiter.record_rate_limit())
    asyncio.run(limiter.record_rate_limit())

    assert limiter.limit == 3


def test_adaptive_rate_limiter_respects_maximum() -> None:
    limiter = AdaptiveRateLimiter(
        AdaptiveRateLimit(
            max_concurrency=100,
            initial_concurrency=90,
            success_window_requests=1,
        )
    )

    asyncio.run(limiter.record_success())

    assert limiter.limit == 100
