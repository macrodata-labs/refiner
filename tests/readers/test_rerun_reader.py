from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import fsspec
import numpy as np
import pytest

import refiner as mdr
from refiner.pipeline import Row
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.sinks.rerun import RerunSink
from refiner.pipeline.sinks.rerun import _sendable_dynamic_table, _sendable_static_table
from refiner.pipeline.sources.readers.rerun import RerunReader, RerunRecording

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
            include_recording=True,
            timelines=("frame",),
            fps=30.0,
        ).take(1)[0],
    )

    recording = row["rerun"]
    assert recording.recording_id == "episode-a"
    assert list(recording.tables) == ["frame"]
    assert recording.tables["frame"].num_rows == 3
    assert row["frames"].num_rows == 3


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

    assert row["frames"].column("action").to_pylist() == [[30.0, 10.0], [40.0, 20.0]]
    assert row["frames"].column("observation.state").to_pylist() == [
        [3.0, 1.0],
        [4.0, 2.0],
    ]
    video = row["observation.images.top"]
    assert video.frame_count == 2


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


def test_write_rerun_uses_source_chunks_without_materialized_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tiny.rrd"
    output = tmp_path / "out-raw-copy"
    _tiny_rrd(source)

    source_iter = mdr.read_rerun(
        str(source),
        materialize_tables=False,
    ).source.read()
    block = cast(list[Row], next(source_iter))
    row = block[0]
    recording = row["rerun"]
    assert recording.use_source_chunks is True
    assert recording.local_source is not None
    assert recording.source_recording_count == 1

    monkeypatch.setattr(
        "refiner.pipeline.sinks.rerun.LocalRrd",
        lambda *args, **kwargs: pytest.fail(
            "writer should reuse a live reader-staged RRD path"
        ),
    )
    monkeypatch.setattr(
        "refiner.pipeline.sinks.rerun._matching_store",
        lambda *args, **kwargs: pytest.fail(
            "unfiltered single-recording copies should not rewrite RRD chunks"
        ),
    )
    sink = RerunSink(str(output))
    sink.write_shard_block("shard-a", block)
    sink.on_shard_complete("shard-a")
    with pytest.raises(StopIteration):
        next(source_iter)

    written = sorted(output.glob("**/*.rrd"))
    assert len(written) == 1
    copied = cast(
        Any, next(mdr.read_rerun(str(written[0]), timelines=("frame",)).source.read())
    )

    assert copied["episode_id"] == "episode-a"
    assert copied["rerun"].tables["frame"].num_rows == 3


def test_write_rerun_reuses_reader_staged_remote_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tiny.rrd"
    output = tmp_path / "out-remote-raw-copy"
    _tiny_rrd(source)

    remote_fs = fsspec.filesystem("memory")
    remote_path = f"/refiner-rerun-test/{tmp_path.name}/tiny.rrd"
    remote_fs.pipe_file(remote_path, source.read_bytes())

    source_iter = mdr.read_rerun(
        (remote_path, remote_fs),
        materialize_tables=False,
    ).source.read()
    block = cast(list[Row], next(source_iter))
    row = block[0]
    recording = row["rerun"]
    assert not recording.source_file.is_local
    assert recording.local_source is not None
    assert recording.local_source.path is not None
    assert recording.local_source.path.exists()
    assert recording.source_recording_count == 1

    monkeypatch.setattr(
        "refiner.pipeline.sinks.rerun.LocalRrd",
        lambda *args, **kwargs: pytest.fail(
            "writer should reuse the reader-staged remote RRD path"
        ),
    )
    sink = RerunSink(str(output))
    sink.write_shard_block("shard-a", block)
    sink.on_shard_complete("shard-a")
    with pytest.raises(StopIteration):
        next(source_iter)
    assert recording.local_source.path is None

    written = sorted(output.glob("**/*.rrd"))
    assert len(written) == 1
    assert written[0].read_bytes() == remote_fs.cat(remote_path)
    copied = cast(
        Any, next(mdr.read_rerun(str(written[0]), timelines=("frame",)).source.read())
    )

    assert copied["episode_id"] == "episode-a"
    assert copied["rerun"].tables["frame"].num_rows == 3


def test_write_rerun_does_not_direct_copy_when_source_may_have_multiple_recordings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tiny.rrd"
    output = tmp_path / "out-not-direct-copy"
    _tiny_rrd(source)

    source_iter = mdr.read_rerun(
        str(source),
        materialize_tables=False,
    ).source.read()
    block = cast(list[Row], next(source_iter))
    row = block[0]
    recording = replace(row["rerun"], source_recording_count=2)

    monkeypatch.setattr(
        "refiner.pipeline.sinks.rerun.shutil.copyfile",
        lambda *args, **kwargs: pytest.fail(
            "multi-recording source rows must use the chunk-selection path"
        ),
    )
    sink = RerunSink(str(output))
    sink.write_shard_block("shard-a", [row.update({"rerun": recording})])
    sink.on_shard_complete("shard-a")
    with pytest.raises(StopIteration):
        next(source_iter)

    written = sorted(output.glob("**/*.rrd"))
    assert len(written) == 1
    copied = cast(
        Any, next(mdr.read_rerun(str(written[0]), timelines=("frame",)).source.read())
    )

    assert copied["episode_id"] == "episode-a"
    assert copied["rerun"].tables["frame"].num_rows == 3


def test_write_rerun_prefers_hardlink_for_local_single_recording_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tiny.rrd"
    output = tmp_path / "out-hardlink-copy"
    _tiny_rrd(source)

    source_iter = mdr.read_rerun(
        str(source),
        materialize_tables=False,
    ).source.read()
    block = cast(list[Row], next(source_iter))
    row = block[0]
    recording = row["rerun"]
    assert recording.local_source is not None
    assert recording.local_source.path is not None
    staged_path = recording.local_source.path

    link_calls: list[tuple[str, str]] = []

    def fake_link(src: str | Path, dst: str | Path) -> None:
        link_calls.append((str(src), str(dst)))
        Path(dst).write_bytes(Path(src).read_bytes())

    monkeypatch.setattr("refiner.pipeline.sinks.rerun.os.link", fake_link)
    monkeypatch.setattr(
        "refiner.pipeline.sinks.rerun.shutil.copyfile",
        lambda *args, **kwargs: pytest.fail("hardlink path should not copy bytes"),
    )

    sink = RerunSink(str(output))
    sink.write_shard_block("shard-a", block)
    sink.on_shard_complete("shard-a")
    with pytest.raises(StopIteration):
        next(source_iter)

    written = sorted(output.glob("**/*.rrd"))
    assert len(written) == 1
    assert link_calls
    assert link_calls[0][0] == str(staged_path)


def test_write_rerun_direct_copy_does_not_require_rerun_sdk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tiny.rrd"
    output = tmp_path / "out-no-rerun-sdk"
    _tiny_rrd(source)

    source_iter = mdr.read_rerun(
        str(source),
        materialize_tables=False,
    ).source.read()
    block = cast(list[Row], next(source_iter))
    row = block[0]
    recording = row["rerun"]
    assert recording.use_source_chunks is True
    assert recording.source_recording_count == 1

    monkeypatch.setattr(
        "refiner.pipeline.sinks.rerun.check_required_dependencies",
        lambda *args, **kwargs: pytest.fail(
            "direct-copy raw writes should not require rerun-sdk"
        ),
    )

    sink = RerunSink(str(output))
    sink.write_shard_block("shard-a", block)
    sink.on_shard_complete("shard-a")
    with pytest.raises(StopIteration):
        next(source_iter)

    written = sorted(output.glob("**/*.rrd"))
    assert len(written) == 1


def test_write_rerun_rejects_timeline_filtered_metadata_only_recording(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tiny.rrd"
    output = tmp_path / "out-timeline-filtered-metadata-only"
    _tiny_rrd(source)

    row = cast(
        Any,
        next(
            mdr.read_rerun(
                str(source),
                timelines=("frame",),
                materialize_tables=False,
            ).source.read()
        ),
    )
    sink = RerunSink(str(output))

    with pytest.raises(ValueError, match="without materialized Rerun table columns"):
        sink.write_shard_block("shard-a", [row])


def test_write_rerun_rejects_segment_id_path_separator_in_filename(
    tmp_path: Path,
) -> None:
    recording = RerunRecording(
        segment_id="episode/5",
        source_path="memory://episode.rrd",
        tables={},
    )
    sink = RerunSink(
        str(tmp_path / "out-segment-id"),
        filename_template="{shard_id}__w{worker_id}/{segment_id}.rrd",
    )

    with pytest.raises(ValueError, match="segment_id must be a single"):
        sink.write_shard_block("shard-a", [DictRow({"rerun": recording})])


def test_write_rerun_rejects_non_row_varying_filename_template(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError, match="requires \\{row_index\\} or \\{segment_id\\}"
    ):
        RerunSink(
            str(tmp_path / "out-overwrite"),
            filename_template="{shard_id}__w{worker_id}.rrd",
        )


def test_write_rerun_rejects_duplicate_rendered_filename(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tiny.rrd"
    output = tmp_path / "out-duplicate"
    _tiny_rrd(source)
    unit = next(mdr.read_rerun(str(source), timelines=("frame",)).source.read())
    assert isinstance(unit, Row)

    sink = RerunSink(
        str(output),
        filename_template="{shard_id}__w{worker_id}/{segment_id}.rrd",
    )
    with pytest.raises(ValueError, match="rendered duplicate output path"):
        sink.write_shard_block("shard-a", [unit, unit])


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


def test_write_rerun_without_footer_rejects_metadata_only_recording(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tiny.rrd"
    output = tmp_path / "out-no-footer-metadata-only"
    _tiny_rrd(source)

    row = cast(
        Any,
        next(
            mdr.read_rerun(
                str(source),
                timelines=("frame",),
                materialize_tables=False,
            ).source.read()
        ),
    )
    sink = RerunSink(str(output), write_footer=False)

    with pytest.raises(ValueError, match="without materialized Rerun table columns"):
        sink.write_shard_block("shard-a", [row])


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
