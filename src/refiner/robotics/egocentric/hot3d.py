from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from refiner.robotics.egocentric.types import HandSide, HaworResult


def load_hot3d_tar_ground_truth(
    tar_path: str | Path,
    *,
    stream_id: str = "214-1",
    fps: float = 30.0,
    hands: tuple[HandSide, ...] = ("left", "right"),
) -> HaworResult:
    """Load HOT3D tar annotations into the HaWoR-style benchmark contract.

    HOT3D stores per-frame wrist transforms as ``T_world_from_wrist``. That is
    enough to score relative wrist actions without reconstructing MANO joints.
    Timestamps are generated from frame index and ``fps`` because benchmark
    predictions are usually produced from a generated constant-FPS RGB video.
    """

    if fps <= 0:
        raise ValueError("fps must be > 0")

    path = Path(tar_path)
    with tarfile.open(path) as archive:
        frame_ids = _frame_ids(archive)
        timestamps = [index / fps for index in range(len(frame_ids))]
        camera = {
            "T_world_camera": [
                _camera_transform(archive, frame_id, stream_id)
                for frame_id in frame_ids
            ],
            "units": "meters",
            "source": "hot3d_ground_truth",
        }
        hand_payloads = {
            side: _hand_payload(archive, frame_ids, side=side, stream_id=stream_id)
            for side in hands
        }

    return HaworResult(
        timestamps=timestamps,
        camera=camera,
        left_hand=hand_payloads.get("left"),
        right_hand=hand_payloads.get("right"),
        metadata={
            "provider": "hot3d_ground_truth",
            "stream_id": stream_id,
            "source_tar": str(path),
        },
    )


def _frame_ids(archive: tarfile.TarFile) -> list[str]:
    ids = sorted(
        name.split(".", 1)[0]
        for name in archive.getnames()
        if name.endswith(".hands.json")
    )
    if not ids:
        raise ValueError("HOT3D tar contains no per-frame hands JSON files")
    return ids


def _camera_transform(
    archive: tarfile.TarFile,
    frame_id: str,
    stream_id: str,
) -> list[list[float]]:
    cameras = _read_json(archive, f"{frame_id}.cameras.json")
    try:
        pose = cameras[stream_id]["T_world_from_camera"]
    except KeyError as exc:
        raise ValueError(
            f"HOT3D frame {frame_id} has no camera stream '{stream_id}'"
        ) from exc
    return _transform_from_quaternion_translation(pose)


def _hand_payload(
    archive: tarfile.TarFile,
    frame_ids: list[str],
    *,
    side: HandSide,
    stream_id: str,
) -> dict[str, Any] | None:
    transforms = []
    mano_pose = []
    confidence = []
    seen = False
    for frame_id in frame_ids:
        hands = _read_json(archive, f"{frame_id}.hands.json")
        hand = hands.get(side)
        if hand is None:
            transforms.append(_nan_transform())
            mano_pose.append([float("nan")] * 15)
            confidence.append(0.0)
            continue
        seen = True
        transforms.append(
            _transform_from_quaternion_translation(
                hand["umetrack_pose"]["T_world_from_wrist"]
            )
        )
        mano_pose.append(list(hand.get("mano_pose", {}).get("thetas", [])))
        confidence.append(
            float(hand.get("visibilities_modeled", {}).get(stream_id, 1.0))
        )
    if not seen:
        return None
    return {
        "T_world_wrist": transforms,
        "mano_pose": mano_pose,
        "confidence": confidence,
        "source": "hot3d_ground_truth",
    }


def _read_json(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    member = archive.extractfile(name)
    if member is None:
        raise ValueError(f"HOT3D tar is missing {name}")
    return json.load(member)


def _transform_from_quaternion_translation(pose: dict[str, Any]) -> list[list[float]]:
    w, x, y, z = [float(value) for value in pose["quaternion_wxyz"]]
    translation = [float(value) for value in pose["translation_xyz"]]
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform.tolist()


def _nan_transform() -> list[list[float]]:
    transform = np.full((4, 4), np.nan, dtype=np.float64)
    transform[3] = [0.0, 0.0, 0.0, 1.0]
    return transform.tolist()


__all__ = ["load_hot3d_tar_ground_truth"]
