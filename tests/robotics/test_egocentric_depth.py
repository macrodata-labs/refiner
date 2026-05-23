from __future__ import annotations

import json
from pathlib import Path

import pytest

from refiner.pipeline.data.row import DictRow
from refiner.robotics.egocentric import (
    estimate_depth_lingbot,
    load_depth_artifact_file,
    validate_depth_payload,
)


def test_load_depth_artifact_file_validates_payload(tmp_path: Path) -> None:
    path = tmp_path / "depth.json"
    payload = {
        "source": "lingbot-depth",
        "timestamps": [0.0, 0.1],
        "metric_depth": {
            "format": "megasam_metric_depth_npz",
            "frame_count": 2,
            "directory": "/tmp/depth",
            "backend": "lingbot",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_depth_artifact_file(path) == payload


def test_validate_depth_payload_rejects_frame_count_mismatch() -> None:
    with pytest.raises(ValueError, match="frame_count"):
        validate_depth_payload(
            {
                "source": "lingbot-depth",
                "timestamps": [0.0],
                "metric_depth": {"format": "npz", "frame_count": 2},
            }
        )


def test_estimate_depth_lingbot_runs_external_command(tmp_path: Path) -> None:
    script = tmp_path / "fake_lingbot.py"
    script.write_text(
        "\n".join(
            [
                "import json, sys",
                "result = sys.argv[sys.argv.index('--result') + 1]",
                "payload = {",
                "  'source': 'fake_lingbot',",
                "  'timestamps': [0.0],",
                "  'metric_depth': {'format': 'npz', 'frame_count': 1}",
                "}",
                "open(result, 'w').write(json.dumps(payload))",
            ]
        ),
        encoding="utf-8",
    )

    mapper = estimate_depth_lingbot(
        command=["python", str(script), "--result", "{result_path}"],
        output_root=tmp_path,
    )
    row = mapper(DictRow({"file_path": "/tmp/video.mp4"}))

    assert row["depth"]["source"] == "fake_lingbot"
    assert Path(row["depth/output_dir"]).exists()
