from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import refiner as mdr
from refiner.pipeline import Row
from refiner.pipeline.sinks.rerun import RerunSink
from refiner.pipeline.sinks.rerun import _sendable_dynamic_table, _sendable_static_table

pytest.importorskip("rerun")


def _tiny_rrd(path: Path) -> None:
    import rerun as rr

    rr.init("refiner_rerun_test", recording_id="episode-a")
    rr.save(path)
    frames = np.arange(3)
    rr.send_columns(
        "/action/x",
        indexes=[rr.TimeColumn("frame", sequence=frames)],
        columns=rr.Scalars.columns(scalars=np.asarray([1.0, 2.0, 3.0])),
    )
    rr.send_columns(
        "/action_extra/y",
        indexes=[rr.TimeColumn("frame", sequence=frames)],
        columns=rr.Scalars.columns(scalars=np.asarray([10.0, 20.0, 30.0])),
    )
    rr.send_columns(
        "/observation/state/y",
        indexes=[rr.TimeColumn("frame", sequence=frames)],
        columns=rr.Scalars.columns(scalars=np.asarray([4.0, 5.0, 6.0])),
    )


def _sparse_rrd(path: Path) -> None:
    import rerun as rr

    rr.init("refiner_rerun_sparse_test", recording_id="episode-sparse")
    rr.save(path)
    rr.send_columns(
        "/action/x",
        indexes=[],
        columns=rr.SeriesLines.columns(names=["x"]),
    )
    rr.send_columns(
        "/action/x",
        indexes=[rr.TimeColumn("frame", sequence=np.asarray([0, 1, 2]))],
        columns=rr.Scalars.columns(scalars=np.asarray([1.0, 2.0, 3.0])),
    )
    rr.send_columns(
        "/action/y",
        indexes=[rr.TimeColumn("frame", sequence=np.asarray([1]))],
        columns=rr.Scalars.columns(scalars=np.asarray([9.0])),
    )


def _custom_robotics_rrd(path: Path) -> None:
    import rerun as rr
    from PIL import Image

    rr.init("refiner_rerun_custom_robotics_test", recording_id="episode-custom")
    rr.save(path)
    frames = np.arange(2)
    rr.send_columns(
        "/robot/actions/z",
        indexes=[rr.TimeColumn("frame", sequence=frames)],
        columns=rr.Scalars.columns(scalars=np.asarray([30.0, 40.0])),
    )
    rr.send_columns(
        "/robot/actions/a",
        indexes=[rr.TimeColumn("frame", sequence=frames)],
        columns=rr.Scalars.columns(scalars=np.asarray([10.0, 20.0])),
    )
    rr.send_columns(
        "/robot/state/b",
        indexes=[rr.TimeColumn("frame", sequence=frames)],
        columns=rr.Scalars.columns(scalars=np.asarray([1.0, 2.0])),
    )
    rr.send_columns(
        "/robot/state/a",
        indexes=[rr.TimeColumn("frame", sequence=frames)],
        columns=rr.Scalars.columns(scalars=np.asarray([3.0, 4.0])),
    )
    blobs: list[bytes] = []
    for color in ((1, 2, 3), (4, 5, 6)):
        image = Image.new("RGB", (1, 1), color=color)
        out = BytesIO()
        image.save(out, format="PNG")
        blobs.append(out.getvalue())
    rr.send_columns(
        "/robot/cameras/top",
        indexes=[rr.TimeColumn("frame", sequence=frames)],
        columns=rr.EncodedImage.columns(
            blob=blobs,
            media_type=["image/png", "image/png"],
        ),
    )


def test_read_rerun_recording_preserves_timeline_table(tmp_path: Path) -> None:
    rrd = tmp_path / "tiny.rrd"
    _tiny_rrd(rrd)

    unit = next(mdr.read_rerun(str(rrd), timelines=("frame",)).source.read())
    assert isinstance(unit, Row)
    row = cast(Any, unit)
    recording = row["rerun"]

    assert row["episode_id"] == "episode-a"
    assert list(recording.tables) == ["frame"]
    assert recording.tables["frame"].num_rows == 3
    assert row["file_path"] == str(rrd)


def test_read_rerun_recording_preserves_sparse_rows(tmp_path: Path) -> None:
    rrd = tmp_path / "sparse.rrd"
    _sparse_rrd(rrd)

    row = cast(Any, next(mdr.read_rerun(str(rrd), timelines=("frame",)).source.read()))
    table = row["rerun"].tables["frame"].table

    assert table.num_rows == 3
    assert table.column("frame").to_pylist() == [0, 1, 2]


def test_read_rerun_robotics_mode_converts_to_robot_row(tmp_path: Path) -> None:
    rrd = tmp_path / "tiny.rrd"
    _tiny_rrd(rrd)

    row = cast(
        Any,
        mdr.read_rerun(str(rrd), output="robotics", fps=30.0)
        .to_robot_rows(
            episode_id_key="episode_id",
            nested_frames_key="frames",
            fps=30.0,
        )
        .take(1)[0],
    )

    assert "rerun" not in row
    assert row.episode_id == "episode-a"
    assert row.num_frames == 3
    assert row.actions.to_pylist() == [[1.0], [2.0], [3.0]]
    assert row.states.to_pylist() == [[4.0], [5.0], [6.0]]


def test_read_rerun_robotics_mode_without_recording_skips_recording_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rrd = tmp_path / "tiny.rrd"
    _tiny_rrd(rrd)

    monkeypatch.setattr(
        "refiner.pipeline.sources.readers.rerun._recording_entries",
        lambda *args, **kwargs: pytest.fail(
            "robotics rows without recording payload do not need store metadata"
        ),
    )

    row = cast(
        Any,
        mdr.read_rerun(str(rrd), output="robotics", fps=30.0).take(1)[0],
    )

    assert "rerun" not in row
    assert row["frames"].num_rows == 3


def test_read_rerun_robotics_mode_respects_explicit_selections(
    tmp_path: Path,
) -> None:
    rrd = tmp_path / "custom.rrd"
    _custom_robotics_rrd(rrd)

    row = cast(
        Any,
        mdr.read_rerun(
            str(rrd),
            output="robotics",
            actions=("/robot/actions/z", "/robot/actions/a"),
            states={
                "first": "/robot/state/a",
                "second": "/robot/state/b",
            },
            videos={"observation.images.top": "/robot/cameras/top"},
            fps=5.0,
        ).take(1)[0],
    )

    assert row["frames"].column("action").to_pylist() == [[30.0, 10.0], [40.0, 20.0]]
    assert row["frames"].column("observation.state").to_pylist() == [
        [3.0, 1.0],
        [4.0, 2.0],
    ]
    video = row["observation.images.top"]
    assert video.frame_count == 2
    assert [frame[0, 0].tolist() for frame in video.iter_frame_arrays()] == [
        [1, 2, 3],
        [4, 5, 6],
    ]


def test_read_rerun_robotics_mode_writes_lerobot(tmp_path: Path) -> None:
    rrd = tmp_path / "tiny.rrd"
    out = tmp_path / "lerobot"
    _tiny_rrd(rrd)

    (
        mdr.read_rerun(str(rrd), output="robotics", fps=30.0, robot_type="testbot")
        .to_robot_rows(
            episode_id_key="episode_id",
            nested_frames_key="frames",
            fps=30.0,
            robot_type="testbot",
        )
        .write_lerobot(str(out), max_video_prepare_in_flight=1)
        .launch_local(
            name="rerun-to-lerobot-test",
            num_workers=1,
            rundir=str(tmp_path / "run"),
        )
    )

    row = cast(Any, mdr.read_lerobot(str(out)).take(1)[0])
    assert row.episode_id == "episode-a"
    assert row.num_frames == 3
    assert row.actions.to_pylist() == [[1.0], [2.0], [3.0]]
    assert row.states.to_pylist() == [[4.0], [5.0], [6.0]]


def test_write_rerun_roundtrips_recording_row(tmp_path: Path) -> None:
    source = tmp_path / "tiny.rrd"
    output = tmp_path / "out"
    _tiny_rrd(source)

    unit = next(mdr.read_rerun(str(source), timelines=("frame",)).source.read())
    assert isinstance(unit, Row)
    sink = RerunSink(str(output))
    sink.write_shard_block("shard-a", [unit])
    sink.on_shard_complete("shard-a")

    written = sorted(output.glob("**/*.rrd"))
    assert len(written) == 1
    row = next(mdr.read_rerun(str(written[0]), timelines=("frame",)).source.read())

    assert isinstance(row, Row)
    recording = row["rerun"]
    assert row["episode_id"] == "episode-a"
    assert recording.tables["frame"].num_rows == 3


def test_write_rerun_without_footer_uses_table_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tiny.rrd"
    output = tmp_path / "out-no-footer"
    _tiny_rrd(source)

    unit = next(mdr.read_rerun(str(source), timelines=("frame",)).source.read())
    assert isinstance(unit, Row)

    monkeypatch.setattr(
        "refiner.pipeline.sinks.rerun._write_source_chunks",
        lambda *args, **kwargs: pytest.fail("raw chunk writer cannot disable footers"),
    )
    sink = RerunSink(str(output), write_footer=False)
    sink.write_shard_block("shard-a", [unit])
    sink.on_shard_complete("shard-a")

    written = sorted(output.glob("**/*.rrd"))
    assert len(written) == 1
    row = next(mdr.read_rerun(str(written[0]), timelines=("frame",)).source.read())

    assert isinstance(row, Row)
    assert row["rerun"].tables["frame"].num_rows == 3


def test_write_rerun_table_fallback_separates_static_columns(tmp_path: Path) -> None:
    source = tmp_path / "sparse.rrd"
    _sparse_rrd(source)

    row = cast(
        Any, next(mdr.read_rerun(str(source), timelines=("frame",)).source.read())
    )
    recording = row["rerun"]

    static = _sendable_static_table(recording.static.table)
    dynamic = _sendable_dynamic_table(recording.tables["frame"].table)

    assert "/action/x:SeriesLines:names" in static.column_names
    assert "/action/x:SeriesLines:names" not in dynamic.column_names
    assert "frame" in dynamic.column_names
