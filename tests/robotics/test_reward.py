from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import cast

import pytest

from refiner.robotics.lerobot_format.row import LeRobotRow
from refiner.robotics.reward import _sample_indexes, _sample_video_frames
from refiner.video.decode import DecodedVideoFrame


class EmptyVideo:
    async def iter_frames(self) -> AsyncIterator[DecodedVideoFrame]:
        if False:
            yield


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


def test_sample_video_frames_raises_when_episode_video_has_no_frames() -> None:
    row = cast(
        LeRobotRow,
        SimpleNamespace(
            episode_index=3,
            length=8,
            videos={"observation.images.image": EmptyVideo()},
        ),
    )

    async def collect_frames() -> None:
        _ = [
            frame
            async for frame in _sample_video_frames(
                row,
                video_key="observation.images.image",
                max_frames=8,
            )
        ]

    with pytest.raises(
        ValueError,
        match="episode 3 video 'observation\\.images\\.image' has no frames",
    ):
        asyncio.run(collect_frames())
