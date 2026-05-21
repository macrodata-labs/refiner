from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from refiner.pipeline.data.row import DictRow
from refiner.robotics.egocentric import (
    HaworResult,
    load_hawor_result_file,
    make_relative_actions,
    reconstruct_hands_hawor,
    export_hawor_rerun,
    relative_actions_from_hawor,
)


def _identity_with_x(x: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _hawor_payload() -> dict:
    return {
        "timestamps": [0.0, 0.1, 0.2],
        "camera": {
            "T_world_camera": [
                _identity_with_x(0.0),
                _identity_with_x(0.1),
                _identity_with_x(0.2),
            ]
        },
        "right_hand": {
            "T_world_wrist": [
                _identity_with_x(1.0),
                _identity_with_x(1.2),
                _identity_with_x(1.5),
            ],
            "mano_pose": [[0.0], [1.0], [2.0]],
            "confidence": [0.9, 0.8, 0.7],
        },
    }


def test_hawor_result_validates_series_lengths() -> None:
    payload = _hawor_payload()
    payload["right_hand"]["confidence"] = [0.9]

    with pytest.raises(ValueError, match="right_hand.confidence"):
        HaworResult.from_mapping(payload)


def test_load_hawor_result_file(tmp_path: Path) -> None:
    result_path = tmp_path / "hawor_result.json"
    result_path.write_text(json.dumps(_hawor_payload()), encoding="utf-8")

    result = load_hawor_result_file(result_path)

    assert result.timestamps == [0.0, 0.1, 0.2]
    assert result.right_hand is not None
    assert result.right_hand["confidence"] == [0.9, 0.8, 0.7]


def test_relative_actions_from_hawor_uses_wrist_frame_delta() -> None:
    result = HaworResult.from_mapping(_hawor_payload())

    actions = relative_actions_from_hawor(result, hands=("right",))

    assert actions["timestamps"] == [0.0, 0.1]
    right = actions["hands"]["right"]
    assert right["wrist_delta"][0][0][3] == pytest.approx(0.2)
    assert right["wrist_delta"][1][0][3] == pytest.approx(0.3)
    assert right["mano_target"] == [[1.0], [2.0]]
    assert right["confidence"] == [0.9, 0.8]


def test_make_relative_actions_updates_row() -> None:
    row = DictRow({"hawor": _hawor_payload()})

    updated = make_relative_actions(hands=("right",))(row)

    assert updated["ego_actions"]["hands"]["right"]["wrist_delta"][0][0][3] == (
        pytest.approx(0.2)
    )


def test_reconstruct_hands_hawor_runs_external_command(tmp_path: Path) -> None:
    script = tmp_path / "fake_hawor.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import pathlib",
                "import sys",
                "out = pathlib.Path(sys.argv[sys.argv.index('--result') + 1])",
                "out.write_text(json.dumps({",
                "  'timestamps': [0.0],",
                "  'camera': {'T_world_camera': [[",
                "    [1.0, 0.0, 0.0, 0.0],",
                "    [0.0, 1.0, 0.0, 0.0],",
                "    [0.0, 0.0, 1.0, 0.0],",
                "    [0.0, 0.0, 0.0, 1.0],",
                "  ]]}",
                "}))",
            ]
        ),
        encoding="utf-8",
    )
    row = DictRow({"file_path": "/tmp/video.mp4"})

    updated = reconstruct_hands_hawor(
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

    assert updated["hawor"]["timestamps"] == [0.0]
    assert Path(updated["hawor/output_dir"]).exists()


def test_export_hawor_rerun_logs_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    fake_rerun = SimpleNamespace(
        init=lambda *args, **kwargs: calls.append(("init", args, kwargs)),
        save=lambda *args, **kwargs: calls.append(("save", args, kwargs)),
        log=lambda *args, **kwargs: calls.append(("log", args, kwargs)),
        set_time=lambda *args, **kwargs: calls.append(("set_time", args, kwargs)),
        LineStrips3D=lambda *args, **kwargs: ("LineStrips3D", args, kwargs),
        Points3D=lambda *args, **kwargs: ("Points3D", args, kwargs),
        Scalars=lambda *args, **kwargs: ("Scalars", args, kwargs),
    )
    monkeypatch.setitem(sys.modules, "rerun", fake_rerun)
    result = HaworResult.from_mapping(_hawor_payload())

    output = export_hawor_rerun(
        result,
        output_path=tmp_path / "debug.rrd",
        hands=("right",),
    )

    assert output == tmp_path / "debug.rrd"
    assert ("init", ("refiner_egocentric_hawor",), {}) in calls
    assert any(call[0] == "save" for call in calls)
    assert any(
        call[0] == "log" and call[1][0] == "world/right_hand/wrist_trajectory"
        for call in calls
    )


def test_export_hawor_rerun_logs_video_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeVideoFrameReference:
        def __init__(self, *args, **kwargs):
            calls.append(("VideoFrameReference", args, kwargs))

        @classmethod
        def columns_nanos(cls, nanoseconds):
            calls.append(("columns_nanos", (nanoseconds,), {}))
            return ["video_frame_reference_columns"]

    fake_rerun = SimpleNamespace(
        init=lambda *args, **kwargs: calls.append(("init", args, kwargs)),
        save=lambda *args, **kwargs: calls.append(("save", args, kwargs)),
        log=lambda *args, **kwargs: calls.append(("log", args, kwargs)),
        send_columns=lambda *args, **kwargs: calls.append(
            ("send_columns", args, kwargs)
        ),
        set_time=lambda *args, **kwargs: calls.append(("set_time", args, kwargs)),
        TimeColumn=lambda *args, **kwargs: ("TimeColumn", args, kwargs),
        AssetVideo=lambda *args, **kwargs: ("AssetVideo", args, kwargs),
        VideoFrameReference=FakeVideoFrameReference,
        LineStrips3D=lambda *args, **kwargs: ("LineStrips3D", args, kwargs),
        LineStrips2D=lambda *args, **kwargs: ("LineStrips2D", args, kwargs),
        Points3D=lambda *args, **kwargs: ("Points3D", args, kwargs),
        Points2D=lambda *args, **kwargs: ("Points2D", args, kwargs),
        Scalars=lambda *args, **kwargs: ("Scalars", args, kwargs),
    )
    monkeypatch.setitem(sys.modules, "rerun", fake_rerun)
    monkeypatch.setattr(
        "refiner.robotics.egocentric.rerun._video_size",
        lambda path: (640, 480),
    )
    payload = _hawor_payload()
    payload["metadata"] = {"img_focal": 500.0}
    payload["right_hand"]["joints_world"] = [
        [[0.0, 0.0, 1.0] for _ in range(21)],
        [[0.0, 0.0, 1.0] for _ in range(21)],
        [[0.0, 0.0, 1.0] for _ in range(21)],
    ]
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake mp4")
    result = HaworResult.from_mapping(payload)

    export_hawor_rerun(
        result,
        output_path=tmp_path / "debug.rrd",
        video_path=video_path,
        hands=("right",),
    )

    assert any(call[0] == "send_columns" and call[1][0] == "video" for call in calls)
    assert any(
        call[0] == "log" and call[1][0] == "video/right_hand/joints" for call in calls
    )
