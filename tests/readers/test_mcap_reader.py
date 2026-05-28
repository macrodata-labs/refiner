from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from refiner.pipeline import read_mcap
from refiner.pipeline.sources.readers.mcap import _ros_image_frame
from refiner.pipeline.sources.readers import McapReader
from refiner.robotics.row import RoboticsRow
from refiner.video import VideoFrameArray

mcap_writer = pytest.importorskip("mcap.writer")


def _write_mcap(path: Path) -> None:
    with path.open("wb") as stream:
        writer = mcap_writer.Writer(stream)
        writer.start()
        schema_id = writer.register_schema(
            name="demo.Json",
            encoding="jsonschema",
            data=b'{"type":"object"}',
        )
        joint_channel = writer.register_channel(
            topic="/joint_states",
            message_encoding="json",
            schema_id=schema_id,
        )
        cmd_channel = writer.register_channel(
            topic="/cmd",
            message_encoding="json",
            schema_id=schema_id,
        )
        image_channel = writer.register_channel(
            topic="/image",
            message_encoding="json",
            schema_id=schema_id,
        )
        messages = [
            (joint_channel, 0, b'{"q":[1,2]}', 0, 1),
            (cmd_channel, 100_000_000, b'{"target":[10]}', 100_000_000, 2),
            (image_channel, 0, b'{"frame":[[[1,2,3]]]}', 0, 3),
            (joint_channel, 1_000_000_000, b'{"q":[3,4]}', 1_000_000_000, 4),
            (cmd_channel, 900_000_000, b'{"target":[20]}', 900_000_000, 5),
            (image_channel, 1_000_000_000, b'{"frame":[[[4,5,6]]]}', 1_000_000_000, 6),
            (joint_channel, 2_000_000_000, b'{"q":[5,6]}', 2_000_000_000, 7),
        ]
        for channel_id, log_time, data, publish_time, sequence in messages:
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time,
                data=data,
                publish_time=publish_time,
                sequence=sequence,
            )
        writer.finish()


def _write_marker_mcap(path: Path) -> None:
    with path.open("wb") as stream:
        writer = mcap_writer.Writer(stream)
        writer.start()
        schema_id = writer.register_schema(
            name="demo.Json",
            encoding="jsonschema",
            data=b'{"type":"object"}',
        )
        state_channel = writer.register_channel(
            topic="/state",
            message_encoding="json",
            schema_id=schema_id,
        )
        marker_channel = writer.register_channel(
            topic="/episode_start",
            message_encoding="json",
            schema_id=schema_id,
        )
        messages = [
            (marker_channel, 0, b'{"episode":0}', 0, 1),
            (state_channel, 10, b'{"q":[1]}', 10, 2),
            (state_channel, 20, b'{"q":[2]}', 20, 3),
            (marker_channel, 100, b'{"episode":1}', 100, 4),
            (state_channel, 110, b'{"q":[3]}', 110, 5),
        ]
        for channel_id, log_time, data, publish_time, sequence in messages:
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time,
                data=data,
                publish_time=publish_time,
                sequence=sequence,
            )
        writer.finish()


def _write_out_of_order_marker_mcap(path: Path) -> None:
    with path.open("wb") as stream:
        writer = mcap_writer.Writer(stream)
        writer.start()
        schema_id = writer.register_schema(
            name="demo.Json",
            encoding="jsonschema",
            data=b'{"type":"object"}',
        )
        state_channel = writer.register_channel(
            topic="/state",
            message_encoding="json",
            schema_id=schema_id,
        )
        marker_channel = writer.register_channel(
            topic="/episode_start",
            message_encoding="json",
            schema_id=schema_id,
        )
        messages = [
            (marker_channel, 100, b'{"episode":1}', 100, 1),
            (state_channel, 110, b'{"q":[3]}', 110, 2),
            (marker_channel, 0, b'{"episode":0}', 0, 3),
            (state_channel, 10, b'{"q":[1]}', 10, 4),
            (state_channel, 20, b'{"q":[2]}', 20, 5),
        ]
        for channel_id, log_time, data, publish_time, sequence in messages:
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time,
                data=data,
                publish_time=publish_time,
                sequence=sequence,
            )
        writer.finish()


def _write_sparse_edge_mcap(path: Path) -> None:
    with path.open("wb") as stream:
        writer = mcap_writer.Writer(stream)
        writer.start()
        schema_id = writer.register_schema(
            name="demo.Json",
            encoding="jsonschema",
            data=b'{"type":"object"}',
        )
        event_channel = writer.register_channel(
            topic="/event",
            message_encoding="json",
            schema_id=schema_id,
        )
        state_channel = writer.register_channel(
            topic="/state",
            message_encoding="json",
            schema_id=schema_id,
        )
        messages = [
            (event_channel, 0, b'{"a":1}', 0, 1),
            (event_channel, 1, b'{"b":2}', 1, 2),
            (state_channel, 10, b'{"q":[1]}', 10, 3),
            (state_channel, 10, b'{"q":[2]}', 10, 4),
        ]
        for channel_id, log_time, data, publish_time, sequence in messages:
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time,
                data=data,
                publish_time=publish_time,
                sequence=sequence,
            )
        writer.finish()


def _write_out_of_order_primary_mcap(path: Path) -> None:
    with path.open("wb") as stream:
        writer = mcap_writer.Writer(stream)
        writer.start()
        schema_id = writer.register_schema(
            name="demo.Json",
            encoding="jsonschema",
            data=b'{"type":"object"}',
        )
        state_channel = writer.register_channel(
            topic="/state",
            message_encoding="json",
            schema_id=schema_id,
        )
        target_channel = writer.register_channel(
            topic="/target",
            message_encoding="json",
            schema_id=schema_id,
        )
        messages = [
            (state_channel, 2_000_000_000, b'{"q":[2]}', 2_000_000_000, 1),
            (target_channel, 100_000_000, b'{"u":[0]}', 100_000_000, 2),
            (state_channel, 0, b'{"q":[0]}', 0, 3),
            (target_channel, 1_900_000_000, b'{"u":[2]}', 1_900_000_000, 4),
            (state_channel, 1_000_000_000, b'{"q":[1]}', 1_000_000_000, 5),
            (target_channel, 1_100_000_000, b'{"u":[1]}', 1_100_000_000, 6),
        ]
        for channel_id, log_time, data, publish_time, sequence in messages:
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time,
                data=data,
                publish_time=publish_time,
                sequence=sequence,
            )
        writer.finish()


def test_mcap_reader_defaults_to_sparse_frame_table(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(str(path)).materialize()[0]

    assert row["file_path"] == str(path)
    assert row["message_count"] == 7
    assert row["topics"] == ["/cmd", "/image", "/joint_states"]
    frames = row["frames"]
    assert set(frames.table.column_names) == {
        "frame_index",
        "timestamp",
        "/joint_states.q",
        "/cmd.target",
        "/image.frame",
    }
    assert frames.table.num_rows == 5
    assert frames.column("/joint_states.q").to_pylist() == [
        [1, 2],
        None,
        None,
        [3, 4],
        [5, 6],
    ]


def test_mcap_reader_maps_fields_and_aligns_to_primary(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(
        str(path),
        fields={
            "state": "/joint_states.q",
            "target": "/cmd.target",
        },
        primary="state",
    ).materialize()[0]

    frames = row["frames"]
    assert frames.table.num_rows == 3
    assert frames.column("timestamp").to_pylist() == [0.0, 1.0, 2.0]
    assert frames.column("state").to_pylist() == [[1, 2], [3, 4], [5, 6]]
    assert frames.column("target").to_pylist() == [[10], [20], [20]]
    assert frames.column("mcap.target.skew_ms").to_pylist() == [100.0, -100.0, -1100.0]
    assert row["fps"] == 1.0


def test_mcap_reader_can_align_to_unselected_primary_source(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(
        str(path),
        fields={"target": "/cmd.target"},
        primary="/joint_states.q",
        include_skew=False,
    ).materialize()[0]

    frames = row["frames"]
    assert frames.table.column_names == ["frame_index", "timestamp", "target"]
    assert frames.table.num_rows == 3
    assert frames.column("timestamp").to_pylist() == [0.0, 1.0, 2.0]
    assert frames.column("target").to_pylist() == [[10], [20], [20]]


def test_mcap_reader_sorts_primary_events_before_alignment(tmp_path: Path) -> None:
    path = tmp_path / "out-of-order.mcap"
    _write_out_of_order_primary_mcap(path)

    row = read_mcap(
        str(path),
        fields={"state": "/state.q", "target": "/target.u"},
        primary="state",
        include_skew=False,
    ).materialize()[0]

    frames = row["frames"]
    assert frames.column("timestamp").to_pylist() == [0.0, 1.0, 2.0]
    assert frames.column("state").to_pylist() == [[0], [1], [2]]
    assert frames.column("target").to_pylist() == [[0], [1], [2]]
    assert row["fps"] == 1.0


def test_mcap_reader_builds_video_frame_arrays_for_robot_rows(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = (
        read_mcap(
            str(path),
            fields={"state": "/joint_states.q"},
            videos={"front": "/image.frame"},
            primary="state",
            fps=12,
        )
        .to_robot_rows(
            nested_frames_key="frames",
            state_key="state",
            timestamp_key="timestamp",
            action_key=None,
            video_keys={"observation.images.front": "videos/front"},
            fps_key="fps",
        )
        .materialize()[0]
    )

    robot_row = cast(RoboticsRow, row)
    assert robot_row.num_frames == 3
    assert robot_row.fps == 12
    assert robot_row.states.to_pylist() == [[1, 2], [3, 4], [5, 6]]
    video = robot_row.videos["observation.images.front"]
    assert isinstance(video, VideoFrameArray)
    assert video.frame_count == 3
    assert video.fps == 12
    assert [frame[0, 0].tolist() for frame in video.iter_frame_arrays()] == [
        [1, 2, 3],
        [4, 5, 6],
        [4, 5, 6],
    ]


def test_mcap_reader_omits_video_topics_from_default_fields(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(str(path), videos={"front": "/image.frame"}).materialize()[0]

    assert "/image.frame" not in row["frames"].table.column_names
    assert row["videos"]["front"].frame_count == 2


def test_mcap_reader_filters_topics(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(str(path), topics=["/cmd"]).materialize()[0]

    assert row["message_count"] == 2
    assert row["topics"] == ["/cmd"]
    assert row["frames"].table.column_names == [
        "frame_index",
        "timestamp",
        "/cmd.target",
    ]


def test_mcap_reader_treats_string_topics_as_one_topic(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(str(path), topics="/cmd").materialize()[0]

    assert row["message_count"] == 2
    assert row["topics"] == ["/cmd"]


def test_mcap_reader_treats_string_fields_as_one_source(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(str(path), fields="/cmd.target").materialize()[0]

    assert row["frames"].table.column_names == [
        "frame_index",
        "timestamp",
        "/cmd.target",
    ]
    assert row["frames"].column("/cmd.target").to_pylist() == [[10], [20]]


def test_mcap_reader_rejects_unknown_field_source(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    with pytest.raises(KeyError, match="/missing.value"):
        read_mcap(str(path), fields={"bad": "/missing.value"}).materialize()


def test_mcap_reader_splits_on_time_gaps(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    rows = read_mcap(
        str(path),
        fields={"state": "/joint_states.q"},
        primary="state",
        episode_splitting={"time_gap_s": 0.5},
    ).materialize()

    assert [row["episode_index"] for row in rows] == [0, 1, 2]
    assert [row["frames"].table.num_rows for row in rows] == [1, 1, 1]
    assert [row["frames"].column("state").to_pylist()[0] for row in rows] == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]


def test_mcap_reader_splits_on_marker_topic(tmp_path: Path) -> None:
    path = tmp_path / "markers.mcap"
    _write_marker_mcap(path)

    rows = read_mcap(
        str(path),
        fields={"state": "/state.q"},
        primary="state",
        episode_splitting={"marker_topic": "/episode_start"},
    ).materialize()

    assert [row["episode_index"] for row in rows] == [0, 1]
    assert [row["frames"].column("state").to_pylist() for row in rows] == [
        [[1], [2]],
        [[3]],
    ]


def test_mcap_reader_sorts_marker_events_before_splitting(tmp_path: Path) -> None:
    path = tmp_path / "out-of-order-markers.mcap"
    _write_out_of_order_marker_mcap(path)

    rows = read_mcap(
        str(path),
        fields={"state": "/state.q"},
        primary="state",
        episode_splitting={"marker_topic": "/episode_start"},
    ).materialize()

    assert [row["frames"].column("state").to_pylist() for row in rows] == [
        [[1], [2]],
        [[3]],
    ]


def test_mcap_reader_reads_marker_topic_with_explicit_topic_filter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "markers.mcap"
    _write_marker_mcap(path)

    rows = read_mcap(
        str(path),
        topics="/state",
        fields={"state": "/state.q"},
        primary="state",
        episode_splitting={"marker_topic": "/episode_start"},
    ).materialize()

    assert [row["frames"].column("state").to_pylist() for row in rows] == [
        [[1], [2]],
        [[3]],
    ]


def test_mcap_reader_marker_topic_is_not_a_default_field(tmp_path: Path) -> None:
    path = tmp_path / "markers.mcap"
    _write_marker_mcap(path)

    rows = read_mcap(
        str(path),
        episode_splitting={"marker_topic": "/episode_start"},
    ).materialize()

    assert rows[0]["frames"].table.column_names == [
        "frame_index",
        "timestamp",
        "/state.q",
    ]


def test_mcap_reader_default_fields_union_optional_keys(tmp_path: Path) -> None:
    path = tmp_path / "sparse-edge.mcap"
    _write_sparse_edge_mcap(path)

    row = read_mcap(str(path), topics="/event").materialize()[0]

    frames = row["frames"]
    assert set(frames.table.column_names) == {
        "frame_index",
        "timestamp",
        "/event.a",
        "/event.b",
    }
    assert frames.column("/event.a").to_pylist() == [1, None]
    assert frames.column("/event.b").to_pylist() == [None, 2]


def test_mcap_reader_sparse_mode_preserves_duplicate_timestamps(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sparse-edge.mcap"
    _write_sparse_edge_mcap(path)

    row = read_mcap(str(path), topics="/state").materialize()[0]

    frames = row["frames"]
    assert frames.column("timestamp").to_pylist() == [1e-08, 1e-08]
    assert frames.column("/state.q").to_pylist() == [[1], [2]]


def test_mcap_reader_decodes_ros_image_stride_and_four_channels() -> None:
    rgba = _ros_image_frame(
        {
            "height": 2,
            "width": 1,
            "encoding": "rgba8",
            "step": 6,
            "data": bytes([1, 2, 3, 4, 99, 99, 5, 6, 7, 8, 99, 99]),
        }
    )
    bgra = _ros_image_frame(
        {
            "height": 1,
            "width": 1,
            "encoding": "bgra8",
            "step": 4,
            "data": bytes([1, 2, 3, 4]),
        }
    )

    assert rgba.tolist() == [[[1, 2, 3]], [[5, 6, 7]]]
    assert bgra.tolist() == [[[3, 2, 1]]]


def test_mcap_reader_rejects_unknown_episode_splitting(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    with pytest.raises(ValueError, match="episode_splitting must be"):
        read_mcap(str(path), episode_splitting="marker")


def test_mcap_reader_plans_files_atomically(tmp_path: Path) -> None:
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    _write_mcap(first)
    _write_mcap(second)

    reader = McapReader([str(first), str(second)], target_shard_bytes=1)

    assert len(reader.list_shards()) == 2
