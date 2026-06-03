from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import refiner as mdr
from refiner.pipeline.data import datatype
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.row import Row
from refiner.robotics.hand_tracking import track_hands
from refiner.robotics.row import _robot_row_converter
from refiner.video import VideoFrameArray


def test_track_hands_runs_episode_batch_map(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    class EpisodeInput:
        def __init__(self, *, frames):
            self.frames = list(frames)

    class HandTrackingConfig:
        def __init__(self, *, hand_reconstruction=None):
            self.hand_reconstruction = hand_reconstruction

    class HaworReconstructionConfig:
        pass

    class Result:
        def __init__(self, index: int):
            self.index = index

        def to_dict(self) -> dict[str, Any]:
            return {
                "episode": self.index,
                "hands_world": [
                    {
                        "handedness": "right",
                        "joints_world": [[float(self.index), 0.0, 0.0]],
                    }
                ],
            }

    class HandTrackingPipeline:
        def __init__(self, config):
            seen["config_type"] = type(config).__name__

        def predict_episodes(self, episodes):
            seen["frame_counts"] = [len(episode.frames) for episode in episodes]
            seen["frame_shapes"] = [
                episode.frames[0].to_ndarray(format="rgb24").shape
                for episode in episodes
            ]
            return [Result(index) for index, _ in enumerate(episodes)]

    log_messages: list[str] = []

    class FakeLogger:
        def info(self, message: str) -> None:
            log_messages.append(message)

    metrics: list[tuple[str, float | int, str | None]] = []

    def log_throughput(
        self,
        label: str,
        value: float | int,
        *,
        unit: str | None = None,
    ) -> None:
        del self
        metrics.append((label, value, unit))

    fake_egovision = ModuleType("egovision")
    setattr(fake_egovision, "EpisodeInput", EpisodeInput)
    setattr(fake_egovision, "HandTrackingConfig", HandTrackingConfig)
    setattr(fake_egovision, "HandTrackingPipeline", HandTrackingPipeline)
    setattr(fake_egovision, "HaworReconstructionConfig", HaworReconstructionConfig)
    monkeypatch.setitem(sys.modules, "egovision", fake_egovision)
    monkeypatch.setattr("refiner.robotics.hand_tracking.logger", FakeLogger())
    monkeypatch.setattr(Row, "log_throughput", log_throughput)

    frames = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    to_robot_row = _robot_row_converter(
        video_keys={"video": "video"},
    )
    rows = [
        to_robot_row(DictRow({"video": VideoFrameArray(frames, fps=10)})),
        to_robot_row(DictRow({"video": VideoFrameArray(frames + 1, fps=10)})),
    ]
    batch_fn = track_hands()

    out = list(batch_fn(rows))

    assert seen == {
        "config_type": "HandTrackingConfig",
        "frame_counts": [2, 2],
        "frame_shapes": [(4, 5, 3), (4, 5, 3)],
    }
    assert log_messages[0] == "Initializing ego-vision hand tracking models"
    assert log_messages[1].startswith("Initialized ego-vision hand tracking models in ")
    assert metrics == [
        ("egovision_frames_decoded", 1, "frames"),
        ("egovision_frames_decoded", 1, "frames"),
        ("egovision_frames_decoded", 1, "frames"),
        ("egovision_frames_decoded", 1, "frames"),
        ("frames_processed", 1, "frames"),
        ("egovision_episodes_processed", 1, "episodes"),
        ("frames_processed", 1, "frames"),
        ("egovision_episodes_processed", 1, "episodes"),
    ]
    assert out[0]["hand_tracking"]["episode"] == 0
    assert out[1]["hand_tracking"]["hands_world"][0]["joints_world"] == [
        [1.0, 0.0, 0.0]
    ]


def test_track_hands_is_available_from_robotics_namespace() -> None:
    pipeline = mdr.from_items([]).batch_map(
        mdr.robotics.track_hands(),
        batch_size=2,
    )

    assert pipeline.pipeline_steps[-1].op_name == "batch_map"


def test_track_hands_runs_from_parquet_robotics_rows(tmp_path, monkeypatch) -> None:
    seen: dict[str, Any] = {}

    class EpisodeInput:
        def __init__(self, *, frames):
            self.frames = list(frames)

    class HandTrackingConfig:
        def __init__(self, *, hand_reconstruction=None):
            self.hand_reconstruction = hand_reconstruction

    class HaworReconstructionConfig:
        pass

    class Result:
        def __init__(self, frame_count: int):
            self.frame_count = frame_count

        def to_dict(self) -> dict[str, Any]:
            return {
                "hands_world": {
                    "right": {
                        "confidence": [1.0] * self.frame_count,
                        "joints_world": [
                            [[float(index), 0.0, 0.0]]
                            for index in range(self.frame_count)
                        ],
                    },
                },
            }

    class HandTrackingPipeline:
        def __init__(self, config):
            seen["config_type"] = type(config).__name__

        def predict_episodes(self, episodes):
            seen["episode_count"] = len(episodes)
            seen["frame_counts"] = [len(episode.frames) for episode in episodes]
            seen["frame_shapes"] = [
                episode.frames[0].to_ndarray(format="rgb24").shape
                for episode in episodes
            ]
            return [Result(len(episode.frames)) for episode in episodes]

    fake_egovision = ModuleType("egovision")
    setattr(fake_egovision, "EpisodeInput", EpisodeInput)
    setattr(fake_egovision, "HandTrackingConfig", HandTrackingConfig)
    setattr(fake_egovision, "HandTrackingPipeline", HandTrackingPipeline)
    setattr(fake_egovision, "HaworReconstructionConfig", HaworReconstructionConfig)
    monkeypatch.setitem(sys.modules, "egovision", fake_egovision)

    frames = np.arange(3 * 4 * 5 * 3, dtype=np.uint8).reshape(3, 4, 5, 3)
    table = pa.Table.from_pylist(
        [
            {
                "episode_id": "episode-1",
                "camera": frames.tolist(),
            }
        ],
    )
    parquet_path = tmp_path / "episodes.parquet"
    pq.write_table(table, parquet_path)

    rows = (
        mdr.read_parquet(
            str(parquet_path),
            dtypes={"camera": datatype.video_frame_array()},
        )
        .to_robot_rows(
            episode_id_key="episode_id",
            fps=12,
            video_keys={"video": "camera"},
        )
        .batch_map(
            mdr.robotics.track_hands(video_key="video"),
            batch_size=2,
        )
        .materialize()
    )

    assert seen == {
        "config_type": "HandTrackingConfig",
        "episode_count": 1,
        "frame_counts": [3],
        "frame_shapes": [(4, 5, 3)],
    }
    assert len(rows) == 1
    assert rows[0]["hand_tracking"]["hands_world"]["right"]["confidence"] == [
        1.0,
        1.0,
        1.0,
    ]
