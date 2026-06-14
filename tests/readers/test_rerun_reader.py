from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import refiner as mdr
from refiner.pipeline import Row
from refiner.pipeline.sinks.rerun import RerunSink

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
        "/observation/state/y",
        indexes=[rr.TimeColumn("frame", sequence=frames)],
        columns=rr.Scalars.columns(scalars=np.asarray([4.0, 5.0, 6.0])),
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
