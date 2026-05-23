from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

HandSide = Literal["left", "right"]


@dataclass(frozen=True, slots=True)
class EgocentricRecording:
    """Composable egocentric reconstruction state.

    Each stage adds one field instead of hiding the whole reconstruction behind
    one research backend. Hand payloads intentionally remain dictionaries
    because MANO, joints, and confidence fields vary across upstream methods.
    """

    timestamps: list[float]
    video_path: str | None = None
    metadata: dict[str, Any] | None = None
    depth: dict[str, Any] | None = None
    camera: dict[str, Any] | None = None
    hands_camera: dict[HandSide, dict[str, Any]] | None = None
    hands_world: dict[HandSide, dict[str, Any]] | None = None

    @classmethod
    def from_hawor(
        cls, result: "HaworResult", *, video_path: str | None = None
    ) -> "EgocentricRecording":
        hands_camera: dict[HandSide, dict[str, Any]] = {}
        hands_world: dict[HandSide, dict[str, Any]] = {}

        for side in ("left", "right"):
            hand = result.hand(side)
            if hand is None:
                continue
            camera_payload = {
                key: value
                for key, value in hand.items()
                if key.startswith("T_camera_")
                or key.endswith("_camera")
                or key in {"joints_camera", "mano_pose", "mano_shape", "confidence"}
            }
            world_payload = {
                key: value
                for key, value in hand.items()
                if key.startswith("T_world_")
                or key.endswith("_world")
                or key in {"joints_world", "mano_pose", "mano_shape", "confidence"}
            }
            if camera_payload:
                hands_camera[side] = camera_payload
            if world_payload:
                hands_world[side] = world_payload

        return cls(
            timestamps=list(result.timestamps),
            video_path=video_path,
            metadata=None if result.metadata is None else dict(result.metadata),
            camera=dict(result.camera),
            hands_camera=hands_camera or None,
            hands_world=hands_world or None,
        )

    def replace(self, **changes: Any) -> "EgocentricRecording":
        return replace(self, **changes)

    def to_hawor_result(self) -> "HaworResult":
        camera = self.camera
        if camera is None:
            raise ValueError(
                "EgocentricRecording requires camera to export HaWoR result"
            )

        hands: dict[HandSide, dict[str, Any] | None] = {"left": None, "right": None}
        for side in ("left", "right"):
            payload: dict[str, Any] = {}
            if self.hands_camera and side in self.hands_camera:
                payload.update(self.hands_camera[side])
            if self.hands_world and side in self.hands_world:
                payload.update(self.hands_world[side])
            if payload:
                hands[side] = payload

        return HaworResult(
            timestamps=list(self.timestamps),
            camera=dict(camera),
            left_hand=hands["left"],
            right_hand=hands["right"],
            metadata=None if self.metadata is None else dict(self.metadata),
        )

    def validate(self) -> None:
        frame_count = len(self.timestamps)
        if frame_count == 0:
            raise ValueError("Egocentric recording requires non-empty timestamps")
        if self.camera is not None:
            _validate_series_length(
                self.camera,
                "T_world_camera",
                frame_count,
                owner="camera",
                required=False,
                prefix="Egocentric recording",
            )
        for owner, hands in (
            ("hands_camera", self.hands_camera),
            ("hands_world", self.hands_world),
        ):
            if hands is None:
                continue
            for side, hand in hands.items():
                for key in (
                    "T_camera_wrist",
                    "T_world_wrist",
                    "joints_camera",
                    "joints_world",
                    "mano_pose",
                    "confidence",
                ):
                    _validate_series_length(
                        hand,
                        key,
                        frame_count,
                        owner=f"{owner}.{side}",
                        required=False,
                        prefix="Egocentric recording",
                    )


@dataclass(frozen=True, slots=True)
class HaworResult:
    """Normalized HaWoR reconstruction payload.

    Refiner treats HaWoR as an external reconstruction provider. This dataclass
    captures the small, stable contract downstream transforms need.
    """

    timestamps: list[float]
    camera: dict[str, Any]
    left_hand: dict[str, Any] | None = None
    right_hand: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "HaworResult":
        timestamps = [float(value) for value in payload.get("timestamps", [])]
        camera = _require_mapping(payload, "camera")
        left_hand = _optional_mapping(payload, "left_hand")
        right_hand = _optional_mapping(payload, "right_hand")
        metadata = _optional_mapping(payload, "metadata")
        result = cls(
            timestamps=timestamps,
            camera=dict(camera),
            left_hand=None if left_hand is None else dict(left_hand),
            right_hand=None if right_hand is None else dict(right_hand),
            metadata=None if metadata is None else dict(metadata),
        )
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "timestamps": self.timestamps,
            "camera": self.camera,
        }
        if self.left_hand is not None:
            payload["left_hand"] = self.left_hand
        if self.right_hand is not None:
            payload["right_hand"] = self.right_hand
        if self.metadata is not None:
            payload["metadata"] = self.metadata
        return payload

    def hand(self, side: HandSide) -> dict[str, Any] | None:
        return self.left_hand if side == "left" else self.right_hand

    def validate(self) -> None:
        frame_count = len(self.timestamps)
        if frame_count == 0:
            raise ValueError("HaWoR result requires non-empty timestamps")
        _validate_series_length(
            self.camera,
            "T_world_camera",
            frame_count,
            owner="camera",
            required=False,
            prefix="HaWoR result",
        )
        for side in ("left", "right"):
            hand = self.hand(side)
            if hand is None:
                continue
            _validate_series_length(
                hand,
                "T_world_wrist",
                frame_count,
                owner=f"{side}_hand",
                required=False,
                prefix="HaWoR result",
            )
            _validate_series_length(
                hand,
                "T_camera_wrist",
                frame_count,
                owner=f"{side}_hand",
                required=False,
                prefix="HaWoR result",
            )
            _validate_series_length(
                hand,
                "mano_pose",
                frame_count,
                owner=f"{side}_hand",
                required=False,
                prefix="HaWoR result",
            )
            _validate_series_length(
                hand,
                "confidence",
                frame_count,
                owner=f"{side}_hand",
                required=False,
                prefix="HaWoR result",
            )


def as_transform_series(value: Any, *, name: str) -> np.ndarray:
    transforms = np.asarray(value, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"{name} must be a sequence of 4x4 transforms")
    return transforms


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"HaWoR result requires mapping field '{key}'")
    return value


def _optional_mapping(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"HaWoR result field '{key}' must be a mapping")
    return value


def _validate_series_length(
    payload: dict[str, Any],
    key: str,
    expected: int,
    *,
    owner: str,
    required: bool,
    prefix: str,
) -> None:
    if key not in payload:
        if required:
            raise ValueError(f"{prefix} {owner} requires '{key}'")
        return
    try:
        length = len(payload[key])
    except TypeError as exc:
        raise ValueError(f"{prefix} {owner}.{key} must be a sequence") from exc
    if length != expected:
        raise ValueError(
            f"{prefix} {owner}.{key} has {length} entries, expected {expected}"
        )
