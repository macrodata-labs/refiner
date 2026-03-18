from __future__ import annotations

import numpy as np

from refiner.media.video.types import VideoFile
from refiner.pipeline.data.row import DictRow
from refiner.robotics import motion_trim


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


def test_motion_trim_rewrites_video_file_bounds() -> None:
    row = DictRow(
        {
            "frames": [
                _frame(frame_index=0, timestamp=0.0, action=0.0, state=0.0),
                _frame(frame_index=1, timestamp=0.1, action=0.0, state=0.0),
                _frame(frame_index=2, timestamp=0.2, action=1.0, state=0.0),
                _frame(frame_index=3, timestamp=0.3, action=1.0, state=0.0),
                _frame(frame_index=4, timestamp=0.4, action=1.0, state=0.0),
            ],
            "video": VideoFile(
                uri="file:///tmp/episode.mp4",
                from_timestamp_s=10.0,
                to_timestamp_s=10.5,
            ),
        }
    )

    trimmed = motion_trim(threshold=0.25, pad_frames=1)(row)

    frames = trimmed["frames"]
    assert len(frames) == 3
    assert [int(frame["frame_index"]) for frame in frames] == [0, 1, 2]
    np.testing.assert_allclose(
        [float(frame["timestamp"]) for frame in frames],
        [0.0, 0.1, 0.2],
    )

    video = trimmed["video"]
    assert isinstance(video, VideoFile)
    assert video.from_timestamp_s == 10.1
    assert video.to_timestamp_s == 10.4


def test_motion_trim_returns_empty_frames_when_no_motion() -> None:
    row = DictRow(
        {
            "frames": [
                _frame(frame_index=0, timestamp=0.0, action=0.0, state=0.0),
                _frame(frame_index=1, timestamp=0.1, action=0.0, state=0.0),
                _frame(frame_index=2, timestamp=0.2, action=0.0, state=0.0),
            ],
            "video": VideoFile(
                uri="file:///tmp/episode.mp4",
                from_timestamp_s=5.0,
                to_timestamp_s=5.3,
            ),
        }
    )

    trimmed = motion_trim(threshold=0.25, pad_frames=0)(row)

    assert trimmed["frames"] == []
    video = trimmed["video"]
    assert isinstance(video, VideoFile)
    assert video.from_timestamp_s == 5.0
    assert video.to_timestamp_s == 5.0
