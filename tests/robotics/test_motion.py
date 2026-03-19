from __future__ import annotations

import numpy as np
import pytest

from refiner.io import DataFolder
from refiner.pipeline.data.row import DictRow
from refiner.robotics import motion_trim
from refiner.robotics.lerobot_format import (
    LeRobotInfo,
    LeRobotMetadata,
    LeRobotRow,
    LeRobotStatsFile,
    LeRobotTasks,
)


def _frame(
    *, frame_index: int, timestamp: float, action: float, state: float
) -> DictRow:
    return DictRow(
        {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "action": np.asarray([action], dtype=np.float32),
            "observation.state": np.asarray([state], dtype=np.float32),
        }
    )


def _row(*, frames: list[DictRow], from_ts: float, to_ts: float) -> LeRobotRow:
    return LeRobotRow(
        DictRow(
            {
                "episode_index": 0,
                "length": len(frames),
                "videos/observation.images.main/chunk_index": 0,
                "videos/observation.images.main/file_index": 0,
                "videos/observation.images.main/from_timestamp": from_ts,
                "videos/observation.images.main/to_timestamp": to_ts,
                "stats/observation.images.main/min": [[[0.0]], [[0.0]], [[0.0]]],
                "stats/observation.images.main/max": [[[1.0]], [[1.0]], [[1.0]]],
                "stats/observation.images.main/mean": [[[0.5]], [[0.5]], [[0.5]]],
                "stats/observation.images.main/std": [[[0.1]], [[0.1]], [[0.1]]],
                "stats/observation.images.main/count": [len(frames)],
            }
        ),
        metadata=LeRobotMetadata(
            info=LeRobotInfo(fps=10, robot_type="mockbot"),
            stats=LeRobotStatsFile({}),
            tasks=LeRobotTasks({0: "pick"}),
        ),
        frames=frames,
        root=DataFolder.resolve("/tmp"),
    )


def test_motion_trim_rewrites_video_file_bounds() -> None:
    row = _row(
        frames=[
            _frame(frame_index=0, timestamp=0.0, action=0.0, state=0.0),
            _frame(frame_index=1, timestamp=0.1, action=0.0, state=0.0),
            _frame(frame_index=2, timestamp=0.2, action=1.0, state=0.0),
            _frame(frame_index=3, timestamp=0.3, action=1.0, state=0.0),
            _frame(frame_index=4, timestamp=0.4, action=1.0, state=0.0),
        ],
        from_ts=10.0,
        to_ts=10.5,
    )

    trimmed = motion_trim(threshold=0.25, pad_frames=1)(row)
    assert isinstance(trimmed, LeRobotRow)

    frames = trimmed["frames"]
    assert frames.num_rows == 4
    assert [int(frame["frame_index"]) for frame in frames] == [0, 1, 2, 3]
    np.testing.assert_allclose(
        [float(frame["timestamp"]) for frame in frames],
        [0.0, 0.1, 0.2, 0.3],
    )

    video = trimmed.videos["observation.images.main"]
    assert video.from_timestamp_s == pytest.approx(10.0)
    assert video.to_timestamp_s == pytest.approx(10.4)
    assert "observation.images.main" not in trimmed.stats


def test_motion_trim_returns_empty_frames_when_no_motion() -> None:
    row = _row(
        frames=[
            _frame(frame_index=0, timestamp=0.0, action=0.0, state=0.0),
            _frame(frame_index=1, timestamp=0.1, action=0.0, state=0.0),
            _frame(frame_index=2, timestamp=0.2, action=0.0, state=0.0),
        ],
        from_ts=5.0,
        to_ts=5.3,
    )

    trimmed = motion_trim(threshold=0.25, pad_frames=0)(row)
    assert isinstance(trimmed, LeRobotRow)

    assert trimmed["frames"] == []
    video = trimmed.videos["observation.images.main"]
    assert video.from_timestamp_s == pytest.approx(5.0)
    assert video.to_timestamp_s == pytest.approx(5.3)
