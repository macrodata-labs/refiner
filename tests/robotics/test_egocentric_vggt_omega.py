from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from refiner.pipeline.data.row import DictRow
from refiner.robotics.egocentric import (
    estimate_geometry_vggt_omega,
    geometry_payload_from_vggt_omega_npz,
    load_vggt_omega_geometry_file,
    write_vggt_omega_geometry_json,
)


def _identity_with_x(x: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = x
    return transform


def test_geometry_payload_from_vggt_omega_npz_inverts_w2c_extrinsics(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "vggt_omega_raw.npz"
    np.savez(
        npz_path,
        extrinsics=np.stack([_identity_with_x(-1.0), _identity_with_x(-2.0)])[:, :3, :],
        intrinsics=np.asarray(
            [
                [[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]],
                [[501.0, 0.0, 320.0], [0.0, 511.0, 240.0], [0.0, 0.0, 1.0]],
            ]
        ),
        depth=np.ones((2, 8, 8), dtype=np.float32),
        depth_conf=np.ones((2, 8, 8), dtype=np.float32),
        timestamps=np.asarray([0.0, 0.5]),
    )

    payload = geometry_payload_from_vggt_omega_npz(npz_path)

    assert payload["source"] == "vggt-omega"
    assert payload["timestamps"] == [0.0, 0.5]
    assert payload["camera"]["T_world_camera"][0][0][3] == pytest.approx(1.0)
    assert payload["camera"]["T_world_camera"][1][0][3] == pytest.approx(2.0)
    assert payload["camera"]["intrinsics"][1][0][0] == pytest.approx(501.0)
    assert payload["depth"]["metric_depth"]["key"] == "depth"
    assert payload["depth"]["metric_depth"]["confidence_key"] == "depth_conf"


def test_write_and_load_vggt_omega_geometry_json(tmp_path: Path) -> None:
    npz_path = tmp_path / "vggt_omega_raw.npz"
    np.savez(
        npz_path,
        T_world_camera=np.stack([_identity_with_x(3.0)]),
        depth=np.ones((1, 2, 2), dtype=np.float32),
    )
    json_path = tmp_path / "geometry.json"

    output = write_vggt_omega_geometry_json(
        npz_path=npz_path,
        output_path=json_path,
    )
    payload = load_vggt_omega_geometry_file(output)

    assert payload["camera"]["T_world_camera"][0][0][3] == pytest.approx(3.0)
    assert payload["depth"]["metric_depth"]["frame_count"] == 1


def test_estimate_geometry_vggt_omega_runs_external_command(tmp_path: Path) -> None:
    script = tmp_path / "fake_vggt.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import pathlib",
                "import sys",
                "out = pathlib.Path(sys.argv[sys.argv.index('--result') + 1])",
                "camera = {'source': 'vggt-omega', 'T_world_camera': [[",
                "  [1.0, 0.0, 0.0, 0.0],",
                "  [0.0, 1.0, 0.0, 0.0],",
                "  [0.0, 0.0, 1.0, 0.0],",
                "  [0.0, 0.0, 0.0, 1.0],",
                "]]}",
                "depth = {'source': 'vggt-omega', 'timestamps': [0.0],",
                "  'metric_depth': {'format': 'fake', 'frame_count': 1}}",
                "out.write_text(json.dumps({",
                "  'source': 'vggt-omega',",
                "  'timestamps': [0.0],",
                "  'camera': camera,",
                "  'depth': depth,",
                "}))",
            ]
        ),
        encoding="utf-8",
    )
    row = DictRow({"file_path": "/tmp/video.mp4"})

    updated = estimate_geometry_vggt_omega(
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

    assert updated["vggt_omega"]["source"] == "vggt-omega"
    assert updated["vggt_omega_camera"]["source"] == "vggt-omega"
    assert updated["vggt_omega_depth"]["metric_depth"]["format"] == "fake"
    assert Path(updated["vggt_omega/output_dir"]).exists()
