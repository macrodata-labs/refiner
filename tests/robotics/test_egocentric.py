from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np

import refiner as mdr
from refiner.pipeline.data.row import DictRow
from refiner.robotics.egocentric import track_hands
from refiner.robotics.row import _robot_row_converter
from refiner.video import VideoFrameArray


def test_track_hands_runs_episode_batch_map(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    class EpisodeInput:
        def __init__(self, *, frames, metadata):
            self.frames = list(frames)
            self.metadata = metadata

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
            seen["metadata"] = [episode.metadata for episode in episodes]
            seen["frame_counts"] = [len(episode.frames) for episode in episodes]
            seen["frame_shapes"] = [
                episode.frames[0].to_ndarray(format="rgb24").shape
                for episode in episodes
            ]
            return [Result(index) for index, _ in enumerate(episodes)]

    fake_egovision = ModuleType("egovision")
    setattr(fake_egovision, "EpisodeInput", EpisodeInput)
    setattr(fake_egovision, "HandTrackingConfig", HandTrackingConfig)
    setattr(fake_egovision, "HandTrackingPipeline", HandTrackingPipeline)
    setattr(fake_egovision, "HaworReconstructionConfig", HaworReconstructionConfig)
    monkeypatch.setitem(sys.modules, "egovision", fake_egovision)

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
        "metadata": [{}, {}],
        "frame_counts": [2, 2],
        "frame_shapes": [(4, 5, 3), (4, 5, 3)],
    }
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
