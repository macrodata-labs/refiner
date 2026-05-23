from __future__ import annotations

from dataclasses import dataclass

import pytest

from refiner.robotics.egocentric import (
    EgocentricRecording,
    HandWorldProjector,
    make_aoe_like_pipeline,
)


def _identity_with_x(x: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


@dataclass(frozen=True)
class _PatchStage:
    values: dict

    def run(self, recording: EgocentricRecording) -> EgocentricRecording:
        return recording.replace(**self.values)


def test_hand_world_projector_preserves_mano_and_projects_camera_hands() -> None:
    recording = EgocentricRecording(
        timestamps=[0.0, 0.1],
        camera={"T_world_camera": [_identity_with_x(1.0), _identity_with_x(2.0)]},
        hands_camera={
            "right": {
                "T_camera_wrist": [_identity_with_x(0.1), _identity_with_x(0.2)],
                "joints_camera": [
                    [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
                    [[0.2, 0.0, 0.0], [0.3, 0.0, 0.0]],
                ],
                "mano_pose": [[0.0], [1.0]],
                "confidence": [0.9, 0.8],
            }
        },
    )

    projected = HandWorldProjector(hands=("right",)).run(recording)

    assert projected.hands_world is not None
    right = projected.hands_world["right"]
    assert right["T_world_wrist"][0][0][3] == pytest.approx(1.1)
    assert right["T_world_wrist"][1][0][3] == pytest.approx(2.2)
    assert right["joints_world"][0][0] == pytest.approx([1.1, 0.0, 0.0])
    assert right["joints_world"][1][1] == pytest.approx([2.3, 0.0, 0.0])
    assert right["mano_pose"] == [[0.0], [1.0]]
    assert right["confidence"] == [0.9, 0.8]


def test_aoe_like_pipeline_runs_depth_camera_hands_then_projection() -> None:
    recording = EgocentricRecording(timestamps=[0.0])
    pipeline = make_aoe_like_pipeline(
        depth=_PatchStage({"depth": {"backend": "lingbot-depth"}}),
        camera=_PatchStage({"camera": {"T_world_camera": [_identity_with_x(1.0)]}}),
        hands=_PatchStage(
            {
                "hands_camera": {
                    "right": {
                        "T_camera_wrist": [_identity_with_x(0.5)],
                        "mano_pose": [[0.25]],
                    }
                }
            }
        ),
        projector=HandWorldProjector(hands=("right",)),
    )

    result = pipeline.run(recording)

    assert result.depth == {"backend": "lingbot-depth"}
    assert result.hands_world is not None
    assert result.hands_world["right"]["T_world_wrist"][0][0][3] == pytest.approx(1.5)
    assert result.hands_world["right"]["mano_pose"] == [[0.25]]


def test_recording_round_trips_to_hawor_result() -> None:
    recording = EgocentricRecording(
        timestamps=[0.0],
        camera={"T_world_camera": [_identity_with_x(1.0)]},
        hands_camera={"right": {"T_camera_wrist": [_identity_with_x(0.2)]}},
        hands_world={"right": {"T_world_wrist": [_identity_with_x(1.2)]}},
    )

    result = recording.to_hawor_result()

    assert result.camera["T_world_camera"][0][0][3] == pytest.approx(1.0)
    assert result.right_hand is not None
    assert result.right_hand["T_camera_wrist"][0][0][3] == pytest.approx(0.2)
    assert result.right_hand["T_world_wrist"][0][0][3] == pytest.approx(1.2)
