from __future__ import annotations

import pytest

import refiner as mdr
from refiner.pipeline.data.row import DictRow
from refiner.robotics import motion_trim


def test_motion_trim_raises_for_top_level_video_columns() -> None:
    row = DictRow(
        {
            "frames": [
                {
                    "frame_index": 0,
                    "timestamp": 0.0,
                    "action": [0.0],
                    "observation.state": [0.0],
                },
                {
                    "frame_index": 1,
                    "timestamp": 0.1,
                    "action": [1.0],
                    "observation.state": [1.0],
                },
            ],
            "observation.images.main": mdr.VideoFile(
                "memory://motion-trim/episode.mp4",
                from_timestamp_s=0.0,
                to_timestamp_s=0.2,
            ),
        }
    )

    with pytest.raises(ValueError, match="does not support top-level video columns"):
        motion_trim()(row)
