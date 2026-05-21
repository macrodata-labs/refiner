from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

HandSide = Literal["left", "right"]


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
            )
            _validate_series_length(
                hand,
                "T_camera_wrist",
                frame_count,
                owner=f"{side}_hand",
                required=False,
            )
            _validate_series_length(
                hand,
                "mano_pose",
                frame_count,
                owner=f"{side}_hand",
                required=False,
            )
            _validate_series_length(
                hand,
                "confidence",
                frame_count,
                owner=f"{side}_hand",
                required=False,
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
) -> None:
    if key not in payload:
        if required:
            raise ValueError(f"HaWoR result {owner} requires '{key}'")
        return
    try:
        length = len(payload[key])
    except TypeError as exc:
        raise ValueError(f"HaWoR result {owner}.{key} must be a sequence") from exc
    if length != expected:
        raise ValueError(
            f"HaWoR result {owner}.{key} has {length} entries, expected {expected}"
        )
