from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from refiner.pipeline.data.row import DictRow
from refiner.robotics.egocentric import (
    camera_payload_from_megasam_npz,
    estimate_camera_megasam,
    load_megasam_trajectory_file,
    write_megasam_trajectory_json,
)


def _identity_with_x(x: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = x
    return transform


def test_camera_payload_from_megasam_npz_converts_cam_c2w(tmp_path: Path) -> None:
    npz_path = tmp_path / "scene_droid.npz"
    np.savez(
        npz_path,
        cam_c2w=np.stack([_identity_with_x(1.0), _identity_with_x(2.0)]),
        intrinsic=np.asarray(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        ),
    )

    payload = camera_payload_from_megasam_npz(npz_path)

    assert payload["source"] == "megasam"
    assert payload["T_world_camera"][0][0][3] == pytest.approx(1.0)
    assert payload["T_world_camera"][1][0][3] == pytest.approx(2.0)
    assert payload["intrinsic"][0][0] == pytest.approx(500.0)


def test_write_and_load_megasam_trajectory_json(tmp_path: Path) -> None:
    npz_path = tmp_path / "scene_droid.npz"
    np.savez(npz_path, cam_c2w=np.stack([_identity_with_x(1.0)]))
    json_path = tmp_path / "trajectory.json"

    output = write_megasam_trajectory_json(npz_path=npz_path, output_path=json_path)
    payload = load_megasam_trajectory_file(output)

    assert payload["T_world_camera"][0][0][3] == pytest.approx(1.0)


def test_estimate_camera_megasam_runs_external_command(tmp_path: Path) -> None:
    script = tmp_path / "fake_megasam.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import pathlib",
                "import sys",
                "out = pathlib.Path(sys.argv[sys.argv.index('--result') + 1])",
                "out.write_text(json.dumps({",
                "  'T_world_camera': [[",
                "    [1.0, 0.0, 0.0, 0.0],",
                "    [0.0, 1.0, 0.0, 0.0],",
                "    [0.0, 0.0, 1.0, 0.0],",
                "    [0.0, 0.0, 0.0, 1.0],",
                "  ]],",
                "  'source': 'fake_megasam'",
                "}))",
            ]
        ),
        encoding="utf-8",
    )
    row = DictRow({"file_path": "/tmp/video.mp4"})

    updated = estimate_camera_megasam(
        command=[
            sys.executable,
            str(script),
            "--video",
            "{video_path}",
            "--result",
            "{result_path}",
        ],
        output_root=tmp_path,
    )(row)

    assert updated["megasam_camera"]["source"] == "fake_megasam"
    assert Path(updated["megasam/output_dir"]).exists()
