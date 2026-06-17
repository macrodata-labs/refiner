from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import refiner as mdr
from refiner.pipeline import Row
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.expressions import col
from refiner.pipeline.sources.readers.rerun import RerunReader
from refiner.robotics.row import RoboticsRow

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


def _sparse_camera_rrd(path: Path) -> None:
    import rerun as rr
    from PIL import Image

    rr.init("refiner_rerun_sparse_camera_test", recording_id="episode-sparse-camera")
    rr.save(path)
    rr.send_columns(
        "/robot/actions/x",
        indexes=[rr.TimeColumn("frame", sequence=np.asarray([0, 1]))],
        columns=rr.Scalars.columns(scalars=np.asarray([1.0, 2.0])),
    )
    out = BytesIO()
    Image.new("RGB", (1, 1), color=(1, 2, 3)).save(out, format="PNG")
    rr.send_columns(
        "/robot/cameras/top",
        indexes=[rr.TimeColumn("frame", sequence=np.asarray([0]))],
        columns=rr.EncodedImage.columns(
            blob=[out.getvalue()],
            media_type=["image/png"],
        ),
    )


def _reserved_video_name_rrd(path: Path) -> None:
    import rerun as rr
    from PIL import Image

    rr.init("refiner_rerun_reserved_video_name_test", recording_id="episode-reserved")
    rr.save(path)
    out = BytesIO()
    Image.new("RGB", (1, 1), color=(1, 2, 3)).save(out, format="PNG")
    rr.send_columns(
        "/frames",
        indexes=[rr.TimeColumn("frame", sequence=np.asarray([0]))],
        columns=rr.EncodedImage.columns(
            blob=[out.getvalue()],
            media_type=["image/png"],
        ),
    )


def _colliding_camera_names_rrd(path: Path) -> None:
    import rerun as rr
    from PIL import Image

    rr.init("refiner_rerun_colliding_camera_names_test", recording_id="episode-cameras")
    rr.save(path)
    frames = np.arange(1)
    blobs: list[bytes] = []
    for color in ((1, 2, 3), (4, 5, 6)):
        out = BytesIO()
        Image.new("RGB", (1, 1), color=color).save(out, format="PNG")
        blobs.append(out.getvalue())
    for entity_path, blob in zip(("/cam/a.b", "/cam/a/b"), blobs, strict=True):
        rr.send_columns(
            entity_path,
            indexes=[rr.TimeColumn("frame", sequence=frames)],
            columns=rr.EncodedImage.columns(
                blob=[blob],
                media_type=["image/png"],
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
    assert recording.source_path == str(rrd)
    assert row["file_path"] == str(rrd)


def test_read_rerun_recording_can_project_before_arrow_conversion(
    tmp_path: Path,
) -> None:
    rrd = tmp_path / "tiny.rrd"
    _tiny_rrd(rrd)

    selected = (
        mdr.read_rerun(str(rrd), timelines=("frame",)).select("episode_id").take(1)[0]
    )
    dropped = mdr.read_rerun(str(rrd), timelines=("frame",)).drop("rerun").take(1)[0]

    assert selected.to_dict() == {"episode_id": "episode-a"}
    assert dropped["episode_id"] == "episode-a"
    assert "rerun" not in dropped


def test_read_rerun_recording_rejects_reserved_file_path_column(
    tmp_path: Path,
) -> None:
    rrd = tmp_path / "tiny.rrd"
    _tiny_rrd(rrd)

    with pytest.raises(ValueError, match="reserved Rerun recording row column 'rerun'"):
        mdr.read_rerun(str(rrd), file_path_column="rerun")


def test_read_rerun_recording_preserves_sparse_rows(tmp_path: Path) -> None:
    rrd = tmp_path / "sparse.rrd"
    _sparse_rrd(rrd)

    row = cast(Any, next(mdr.read_rerun(str(rrd), timelines=("frame",)).source.read()))
    table = row["rerun"].tables["frame"].table

    assert table.num_rows == 3
    assert table.column("frame").to_pylist() == [0, 1, 2]


def test_read_rerun_recording_can_skip_table_materialization(tmp_path: Path) -> None:
    rrd = tmp_path / "tiny.rrd"
    _tiny_rrd(rrd)

    row = cast(
        Any,
        next(
            mdr.read_rerun(
                str(rrd),
                timelines=("frame",),
                materialize_tables=False,
            ).source.read()
        ),
    )
    recording = row["rerun"]

    assert recording.recording_id == "episode-a"
    assert recording.tables == {}
    assert recording.static is None
    assert recording.source_file is not None
    assert recording.timelines == ("frame",)
    assert recording.use_source_chunks is False


def test_read_rerun_recording_without_materialized_tables_scans_metadata_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rrd = tmp_path / "tiny.rrd"
    _tiny_rrd(rrd)
    calls = 0

    def fake_recording_entries(*args: Any, **kwargs: Any) -> list[Any]:
        nonlocal calls
        del args, kwargs
        calls += 1
        return [
            type(
                "Store",
                (),
                {"recording_id": "episode-a", "application_id": "refiner"},
            )()
        ]

    monkeypatch.setattr(
        "refiner.pipeline.sources.readers.rerun._recording_entries",
        fake_recording_entries,
    )

    row = cast(
        Any,
        next(
            mdr.read_rerun(
                str(rrd),
                materialize_tables=False,
            ).source.read()
        ),
    )

    assert isinstance(row, list)
    assert row[0]["episode_id"] == "episode-a"
    assert calls == 1


def test_read_rerun_rejects_ignored_mode_options(tmp_path: Path) -> None:
    rrd = tmp_path / "tiny.rrd"
    _tiny_rrd(rrd)

    with pytest.raises(
        ValueError,
        match="materialize_tables=False is only supported for recording output",
    ):
        mdr.read_rerun(str(rrd), output="robotics", materialize_tables=False)

    with pytest.raises(
        ValueError,
        match="include_recording=False is only supported for robotics output",
    ):
        mdr.read_rerun(str(rrd), output="recording", include_recording=False)
    with pytest.raises(
        ValueError,
        match="Rerun recording output does not use robotics options: primary_timeline",
    ):
        mdr.read_rerun(str(rrd), output="recording", primary_timeline="frame")
    with pytest.raises(
        ValueError,
        match="Rerun recording output does not use robotics options: actions, fps",
    ):
        mdr.read_rerun(
            str(rrd),
            output="recording",
            actions=("/action/x",),
            fps=30.0,
        )


def test_read_rerun_recording_without_materialized_tables_skips_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rrd = tmp_path / "tiny.rrd"
    _tiny_rrd(rrd)

    monkeypatch.setattr(
        "refiner.pipeline.sources.readers.rerun.RerunReader._read_files_with_server",
        lambda *args, **kwargs: pytest.fail(
            "metadata-only recording rows do not need the Rerun server"
        ),
    )

    row = cast(
        Any,
        next(
            mdr.read_rerun(
                str(rrd),
                timelines=("frame",),
                materialize_tables=False,
            ).source.read()
        ),
    )

    assert row["episode_id"] == "episode-a"
    assert row["rerun"].tables == {}


def test_read_rerun_batches_small_files_in_one_staged_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.rrd"
    second = tmp_path / "second.rrd"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    reader = RerunReader(
        [str(first), str(second)],
        target_shard_bytes=1024 * 1024,
    )
    batch_sizes: list[int] = []

    def fake_read_files(self: RerunReader, local_files: Any) -> Any:
        del self
        local_files = tuple(local_files)
        batch_sizes.append(len(local_files))
        for source, local_path, _local_rrd in local_files:
            assert local_path.exists()
            yield DictRow({"source": source.abs_path()})

    monkeypatch.setattr(RerunReader, "_read_files", fake_read_files)

    rows = cast(list[Row], list(reader.read_shard(reader.list_shards()[0])))

    assert batch_sizes == [2]
    assert [row["source"] for row in rows] == [str(first), str(second)]


def test_read_rerun_robotics_mode_converts_to_robot_row(tmp_path: Path) -> None:
    rrd = tmp_path / "tiny.rrd"
    _tiny_rrd(rrd)

    row = cast(
        Any,
        mdr.read_rerun(str(rrd), output="robotics", fps=30.0).take(1)[0],
    )

    assert isinstance(row, RoboticsRow)
    assert "rerun" in row
    assert "frames" not in row
    assert row.episode_id == "episode-a"
    assert row.num_frames == 3
    assert row.actions.to_pylist() == [[1.0], [2.0], [3.0]]
    assert row.states.to_pylist() == [[4.0], [5.0], [6.0]]
    assert row.to_frame_table().column("action").to_pylist() == [
        [1.0],
        [2.0],
        [3.0],
    ]


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
        mdr.read_rerun(
            str(rrd),
            output="robotics",
            include_recording=False,
            fps=30.0,
        ).take(1)[0],
    )

    assert "rerun" not in row
    assert "frames" not in row
    assert row.actions.to_pylist() == [[1.0], [2.0], [3.0]]


def test_read_rerun_robotics_mode_exposes_top_level_fields_to_primitives(
    tmp_path: Path,
) -> None:
    rrd = tmp_path / "tiny.rrd"
    _tiny_rrd(rrd)

    selected = cast(
        Any,
        mdr.read_rerun(str(rrd), output="robotics", fps=30.0)
        .select("action")
        .take(1)[0],
    )
    dropped = cast(
        Any,
        mdr.read_rerun(str(rrd), output="robotics", fps=30.0)
        .drop("observation.state")
        .take(1)[0],
    )
    without_recording = cast(
        Any,
        mdr.read_rerun(str(rrd), output="robotics", fps=30.0).drop("rerun").take(1)[0],
    )
    filtered = cast(
        Any,
        mdr.read_rerun(str(rrd), output="robotics", fps=30.0)
        .filter(col("episode_id") == "episode-a")
        .take(1)[0],
    )
    projected_filtered = cast(
        Any,
        mdr.read_rerun(str(rrd), output="robotics", fps=30.0)
        .drop("rerun")
        .filter(col("episode_id") == "episode-a")
        .take(1)[0],
    )

    assert selected["action"].to_pylist() == [[1.0], [2.0], [3.0]]
    assert "rerun" not in selected
    assert "observation.state" not in dropped
    assert dropped.states is None
    assert dropped["action"].to_pylist() == [[1.0], [2.0], [3.0]]
    assert "rerun" not in without_recording
    assert without_recording.actions.to_pylist() == [[1.0], [2.0], [3.0]]
    assert "rerun" in filtered
    assert filtered.actions.to_pylist() == [[1.0], [2.0], [3.0]]
    assert "rerun" not in projected_filtered
    assert projected_filtered.actions.to_pylist() == [[1.0], [2.0], [3.0]]


def test_read_rerun_robotics_mode_with_recording_includes_recording_payload(
    tmp_path: Path,
) -> None:
    rrd = tmp_path / "tiny.rrd"
    _tiny_rrd(rrd)

    row = cast(
        Any,
        mdr.read_rerun(
            str(rrd),
            output="robotics",
            timelines=("frame",),
            fps=30.0,
        ).take(1)[0],
    )

    recording = row["rerun"]
    assert recording.recording_id == "episode-a"
    assert list(recording.tables) == ["frame"]
    assert recording.tables["frame"].num_rows == 3
    assert "frames" not in row
    assert row.actions.to_pylist() == [[1.0], [2.0], [3.0]]


def test_read_rerun_describe_includes_robotics_metadata(tmp_path: Path) -> None:
    rrd = tmp_path / "tiny.rrd"
    _tiny_rrd(rrd)

    description = mdr.read_rerun(
        str(rrd),
        output="robotics",
        fps=12.5,
        robot_type="testbot",
    ).source.describe()

    assert description["fps"] == 12.5
    assert description["robot_type"] == "testbot"


def test_read_rerun_robotics_rejects_reserved_implicit_video_name(
    tmp_path: Path,
) -> None:
    rrd = tmp_path / "reserved-video.rrd"
    _reserved_video_name_rrd(rrd)

    with pytest.raises(
        ValueError,
        match="Rerun video output names cannot use reserved robotics row columns: frames",
    ):
        mdr.read_rerun(
            str(rrd),
            output="robotics",
            camera_prefix="/",
            fps=30.0,
        ).take(1)


def test_read_rerun_robotics_rejects_derived_video_name_collision(
    tmp_path: Path,
) -> None:
    rrd = tmp_path / "colliding-cameras.rrd"
    _colliding_camera_names_rrd(rrd)

    with pytest.raises(
        ValueError,
        match="derive the same output video name",
    ):
        mdr.read_rerun(
            str(rrd),
            output="robotics",
            camera_prefix="/cam",
            fps=30.0,
        ).take(1)


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

    assert row.actions.to_pylist() == [[30.0, 10.0], [40.0, 20.0]]
    assert row.states.to_pylist() == [
        [3.0, 1.0],
        [4.0, 2.0],
    ]
    video = row["observation.images.top"]
    assert video.frame_count == 2
    assert [frame[0, 0].tolist() for frame in video.iter_frame_arrays()] == [
        [1, 2, 3],
        [4, 5, 6],
    ]


def test_read_rerun_robotics_mode_rejects_sparse_video_stream(
    tmp_path: Path,
) -> None:
    rrd = tmp_path / "sparse-camera.rrd"
    _sparse_camera_rrd(rrd)

    with pytest.raises(
        ValueError,
        match="missing frames on the primary timeline",
    ):
        mdr.read_rerun(
            str(rrd),
            output="robotics",
            timelines=("frame",),
            actions=("/robot/actions/x",),
            videos={"observation.images.top": "/robot/cameras/top"},
            fps=5.0,
        ).take(1)


def test_read_rerun_robotics_mode_with_explicit_timeline_uses_table_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rrd = tmp_path / "custom.rrd"
    _custom_robotics_rrd(rrd)

    monkeypatch.setattr(
        "refiner.pipeline.sources.readers.rerun.RerunReader._timelines",
        lambda *args, **kwargs: pytest.fail(
            "explicit timelines should not require schema timeline discovery"
        ),
    )

    row = cast(
        Any,
        mdr.read_rerun(
            str(rrd),
            output="robotics",
            timelines=("frame",),
            actions=("/robot/actions/z", "/robot/actions/a"),
            states={
                "first": "/robot/state/a",
                "second": "/robot/state/b",
            },
            videos={"observation.images.top": "/robot/cameras/top"},
            fps=5.0,
        ).take(1)[0],
    )

    assert row.actions.to_pylist() == [[30.0, 10.0], [40.0, 20.0]]
    assert row.states.to_pylist() == [
        [3.0, 1.0],
        [4.0, 2.0],
    ]
    video = row["observation.images.top"]
    assert video.frame_count == 2
