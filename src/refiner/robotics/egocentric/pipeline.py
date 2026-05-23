from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from refiner.robotics.egocentric.types import (
    EgocentricRecording,
    HandSide,
    as_transform_series,
)


class EgocentricStage(Protocol):
    """A single egocentric reconstruction stage."""

    def run(self, recording: EgocentricRecording) -> EgocentricRecording: ...


class DepthEstimator(EgocentricStage, Protocol):
    """Adds depth maps or depth artifact references to a recording."""


class CameraTrajectoryEstimator(EgocentricStage, Protocol):
    """Adds camera extrinsics such as ``T_world_camera`` to a recording."""


class HandReconstructor(EgocentricStage, Protocol):
    """Adds camera-space hand reconstruction, usually MANO plus 3D joints."""


@dataclass(frozen=True, slots=True)
class HandWorldProjector:
    """Project camera-space hands into the selected world coordinate frame.

    The projector preserves MANO fields from the hand reconstructor and computes
    world-space wrist transforms and joints from ``T_world_camera``.
    """

    hands: tuple[HandSide, ...] = ("left", "right")

    def run(self, recording: EgocentricRecording) -> EgocentricRecording:
        if recording.camera is None or "T_world_camera" not in recording.camera:
            raise ValueError("HandWorldProjector requires camera.T_world_camera")
        if recording.hands_camera is None:
            raise ValueError("HandWorldProjector requires camera-space hands")

        camera = as_transform_series(
            recording.camera["T_world_camera"],
            name="camera.T_world_camera",
        )
        world_hands = dict(recording.hands_world or {})
        for side in self.hands:
            hand = recording.hands_camera.get(side)
            if hand is None:
                continue
            world_hands[side] = _project_hand(camera, hand, side=side)

        updated = recording.replace(hands_world=world_hands or None)
        updated.validate()
        return updated


@dataclass(frozen=True, slots=True)
class EgocentricPipeline:
    """Run egocentric reconstruction stages in order."""

    stages: Sequence[EgocentricStage]

    def run(self, recording: EgocentricRecording) -> EgocentricRecording:
        current = recording
        current.validate()
        for stage in self.stages:
            current = stage.run(current)
            current.validate()
        return current


def make_aoe_like_pipeline(
    *,
    depth: DepthEstimator,
    camera: CameraTrajectoryEstimator,
    hands: HandReconstructor,
    projector: HandWorldProjector | None = None,
) -> EgocentricPipeline:
    """Build the AoE-style geometry pipeline.

    AoE-like here means depth first, MegaSAM-style trajectory second, HaWoR-like
    camera-space hand reconstruction third, then world-space projection.
    """

    return EgocentricPipeline(
        stages=[
            depth,
            camera,
            hands,
            projector or HandWorldProjector(),
        ]
    )


def _project_hand(
    camera: np.ndarray,
    hand: dict,
    *,
    side: HandSide,
) -> dict:
    output = {
        key: value
        for key, value in hand.items()
        if key in {"mano_pose", "mano_shape", "confidence"}
    }

    if "T_camera_wrist" in hand:
        wrist_camera = as_transform_series(
            hand["T_camera_wrist"],
            name=f"{side}_hand.T_camera_wrist",
        )
        if len(wrist_camera) != len(camera):
            raise ValueError(
                f"{side}_hand.T_camera_wrist has {len(wrist_camera)} entries, "
                f"expected {len(camera)}"
            )
        output["T_world_wrist"] = np.matmul(camera, wrist_camera).tolist()

    if "joints_camera" in hand:
        joints_camera = np.asarray(hand["joints_camera"], dtype=np.float64)
        if joints_camera.ndim != 3 or joints_camera.shape[2] != 3:
            raise ValueError(f"{side}_hand.joints_camera must be shaped TxJx3")
        if len(joints_camera) != len(camera):
            raise ValueError(
                f"{side}_hand.joints_camera has {len(joints_camera)} entries, "
                f"expected {len(camera)}"
            )
        output["joints_world"] = _transform_points(camera, joints_camera).tolist()

    return output


def _transform_points(transforms: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((*points.shape[:2], 1), dtype=np.float64)
    homogeneous = np.concatenate([points, ones], axis=2)
    world = np.einsum("tij,tkj->tki", transforms, homogeneous)
    return world[:, :, :3]


__all__ = [
    "CameraTrajectoryEstimator",
    "DepthEstimator",
    "EgocentricPipeline",
    "EgocentricStage",
    "HandReconstructor",
    "HandWorldProjector",
    "make_aoe_like_pipeline",
]
