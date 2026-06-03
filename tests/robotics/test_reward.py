from __future__ import annotations

import pytest

from refiner.robotics.reward import _sample_indexes


@pytest.mark.parametrize(
    ("length", "max_frames", "expected"),
    [
        (0, 8, ()),
        (3, 8, (0, 1, 2)),
        (10, 1, (9,)),
        (10, 4, (0, 3, 6, 9)),
        (100, 8, (0, 14, 28, 42, 57, 71, 85, 99)),
    ],
)
def test_sample_indexes_selects_evenly_spaced_episode_indexes(
    length: int,
    max_frames: int,
    expected: tuple[int, ...],
) -> None:
    assert _sample_indexes(length, max_frames=max_frames) == expected
