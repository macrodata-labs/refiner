from __future__ import annotations

import json
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from refiner.robotics.egocentric import load_hot3d_tar_ground_truth


def _add_json(archive: tarfile.TarFile, name: str, payload: dict) -> None:
    data = json.dumps(payload).encode()
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, BytesIO(data))


def _camera_payload(x: float) -> dict:
    return {
        "214-1": {
            "T_world_from_camera": {
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation_xyz": [x, 0.0, 0.0],
            },
            "calibration": {
                "image_height": 1408,
                "image_width": 1408,
                "projection_model_type": "CameraModelType.FISHEYE624",
                "projection_params": [609.0, 704.0, 704.0],
            },
        }
    }


def _hands_payload(x: float, *, include_right: bool = True) -> dict:
    payload = {
        "left": {
            "umetrack_pose": {
                "T_world_from_wrist": {
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "translation_xyz": [-x, 0.0, 0.0],
                }
            },
            "mano_pose": {"thetas": [0.0] * 15},
            "visibilities_modeled": {"214-1": 0.5},
        }
    }
    if include_right:
        payload["right"] = {
            "umetrack_pose": {
                "T_world_from_wrist": {
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "translation_xyz": [x, 0.0, 0.0],
                }
            },
            "mano_pose": {"thetas": [x] * 15},
            "visibilities_modeled": {"214-1": 0.9},
        }
    return payload


def test_load_hot3d_tar_ground_truth_uses_world_wrist_trajectory(
    tmp_path: Path,
) -> None:
    tar_path = tmp_path / "clip.tar"
    with tarfile.open(tar_path, "w") as archive:
        _add_json(archive, "000000.cameras.json", _camera_payload(1.0))
        _add_json(archive, "000000.hands.json", _hands_payload(0.1))
        _add_json(archive, "000001.cameras.json", _camera_payload(2.0))
        _add_json(archive, "000001.hands.json", _hands_payload(0.2))
        _add_json(archive, "__hand_shapes.json__", {"mano": [0.0] * 10})

    result = load_hot3d_tar_ground_truth(tar_path, hands=("right",), fps=10.0)

    assert result.timestamps == [0.0, 0.1]
    assert result.camera["T_world_camera"][1][0][3] == pytest.approx(2.0)
    assert result.right_hand is not None
    assert result.right_hand["T_world_wrist"][1][0][3] == pytest.approx(0.2)
    assert result.right_hand["mano_pose"][1] == pytest.approx([0.2] * 15)
    assert result.right_hand["confidence"] == [0.9, 0.9]


def test_load_hot3d_tar_ground_truth_marks_missing_hands_invalid(
    tmp_path: Path,
) -> None:
    tar_path = tmp_path / "clip.tar"
    with tarfile.open(tar_path, "w") as archive:
        _add_json(archive, "000000.cameras.json", _camera_payload(1.0))
        _add_json(archive, "000000.hands.json", _hands_payload(0.1))
        _add_json(archive, "000001.cameras.json", _camera_payload(2.0))
        _add_json(
            archive,
            "000001.hands.json",
            _hands_payload(0.2, include_right=False),
        )

    result = load_hot3d_tar_ground_truth(tar_path, hands=("right",))

    assert result.right_hand is not None
    assert result.right_hand["confidence"] == [0.9, 0.0]
    assert result.right_hand["mano_pose"][1] == pytest.approx(
        [float("nan")] * 15, nan_ok=True
    )
