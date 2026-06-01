from __future__ import annotations

import base64
import json
from io import BufferedReader, BytesIO
from pathlib import Path
from typing import Any, cast

from fsspec.implementations.local import LocalFileSystem
import numpy as np
import pytest

from refiner.pipeline import read_mcap
from refiner.pipeline.sources.readers.mcap import _frame_from_value, _ros_image_frame
from refiner.pipeline.sources.readers import mcap as mcap_reader
from refiner.pipeline.sources.readers import McapReader
from refiner.robotics.row import RoboticsRow
from refiner.video import VideoFrameArray

mcap_writer = pytest.importorskip("mcap.writer")


class _NonSeekableReader:
    def __init__(self, stream: BufferedReader):
        self._stream = stream

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *args):
        return self._stream.__exit__(*args)

    def seekable(self) -> bool:
        return False

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


class _NonSeekableLocalFileSystem(LocalFileSystem):
    def open(
        self,
        path,
        mode="rb",
        block_size=None,
        cache_options=None,
        compression=None,
        **kwargs,
    ):
        return _NonSeekableReader(
            super().open(
                path,
                mode=mode,
                block_size=block_size,
                cache_options=cache_options,
                compression=compression,
                **kwargs,
            )
        )


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


def _write_custom_encoding_mcap(path: Path) -> None:
    with path.open("wb") as stream:
        writer = mcap_writer.Writer(stream)
        writer.start()
        schema_id = writer.register_schema(
            name="demo.Custom",
            encoding="custom",
            data=b"",
        )
        channel = writer.register_channel(
            topic="/custom",
            message_encoding="custom",
            schema_id=schema_id,
        )
        writer.add_message(
            channel_id=channel,
            log_time=0,
            data=b"7",
            publish_time=0,
            sequence=1,
        )
        writer.finish()


def _write_nanosecond_rounded_fps_mcap(path: Path) -> None:
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
        image_channel = writer.register_channel(
            topic="/image",
            message_encoding="json",
            schema_id=schema_id,
        )
        for index, timestamp_ns in enumerate((0, 33_333_333, 66_666_666), start=1):
            writer.add_message(
                channel_id=state_channel,
                log_time=timestamp_ns,
                data=f'{{"q":[{index}]}}'.encode(),
                publish_time=timestamp_ns,
                sequence=index * 2 - 1,
            )
            writer.add_message(
                channel_id=image_channel,
                log_time=timestamp_ns,
                data=f'{{"frame":[[[{index},{index},{index}]]]}}'.encode(),
                publish_time=timestamp_ns,
                sequence=index * 2,
            )
        writer.finish()


def _h264_chunk(rgb: tuple[int, int, int]) -> bytes:
    av = pytest.importorskip("av")

    output = BytesIO()
    container = av.open(output, mode="w", format="h264")
    stream = container.add_stream("libx264", rate=2)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    frame_array = np.zeros((16, 16, 3), dtype=np.uint8)
    frame_array[:] = rgb
    frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return output.getvalue()


def _write_h264_mcap(path: Path) -> None:
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
        video_channel = writer.register_channel(
            topic="/video",
            message_encoding="json",
            schema_id=schema_id,
        )
        messages = [
            (state_channel, 0, b'{"q":[1]}', 0, 1),
            (
                video_channel,
                0,
                json.dumps(
                    {
                        "format": "h264",
                        "data": base64.b64encode(_h264_chunk((255, 0, 0))).decode(
                            "ascii"
                        ),
                    }
                ).encode("utf-8"),
                0,
                2,
            ),
            (state_channel, 500_000_000, b'{"q":[2]}', 500_000_000, 3),
            (
                video_channel,
                500_000_000,
                json.dumps(
                    {
                        "format": "h264",
                        "data": base64.b64encode(_h264_chunk((0, 255, 0))).decode(
                            "ascii"
                        ),
                    }
                ).encode("utf-8"),
                500_000_000,
                4,
            ),
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


def _write_marker_mcap_with_sparse_action(path: Path) -> None:
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
        action_channel = writer.register_channel(
            topic="/action",
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
            (action_channel, 20, b'{"u":[10]}', 20, 3),
            (marker_channel, 100, b'{"episode":1}', 100, 4),
            (state_channel, 110, b'{"q":[2]}', 110, 5),
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


def _write_out_of_order_video_mcap(path: Path) -> None:
    with path.open("wb") as stream:
        writer = mcap_writer.Writer(stream)
        writer.start()
        schema_id = writer.register_schema(
            name="demo.Json",
            encoding="jsonschema",
            data=b'{"type":"object"}',
        )
        image_channel = writer.register_channel(
            topic="/image",
            message_encoding="json",
            schema_id=schema_id,
        )
        messages = [
            (image_channel, 1_000_000_000, b'{"frame":[[[4,5,6]]]}', 1_000_000_000, 1),
            (image_channel, 0, b'{"frame":[[[1,2,3]]]}', 0, 2),
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


def _write_duplicate_video_timestamp_mcap(path: Path) -> None:
    with path.open("wb") as stream:
        writer = mcap_writer.Writer(stream)
        writer.start()
        schema_id = writer.register_schema(
            name="demo.Json",
            encoding="jsonschema",
            data=b'{"type":"object"}',
        )
        image_channel = writer.register_channel(
            topic="/image",
            message_encoding="json",
            schema_id=schema_id,
        )
        messages = [
            (image_channel, 0, b'{"frame":[[[1,2,3]]]}', 0, 1),
            (image_channel, 0, b'{"frame":[[[4,5,6]]]}', 0, 2),
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


def _write_late_video_mcap(path: Path) -> None:
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
        image_channel = writer.register_channel(
            topic="/image",
            message_encoding="json",
            schema_id=schema_id,
        )
        messages = [
            (state_channel, 0, b'{"q":[1]}', 0, 1),
            (image_channel, 1_000_000_000, b'{"frame":[[[4,5,6]]]}', 1_000_000_000, 2),
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


def _write_marker_mcap_with_video_only_episode(path: Path) -> None:
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
        image_channel = writer.register_channel(
            topic="/image",
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
            (image_channel, 10, b'{"frame":[[[1,2,3]]]}', 10, 3),
            (marker_channel, 100, b'{"episode":1}', 100, 4),
            (image_channel, 110, b'{"frame":[[[4,5,6]]]}', 110, 5),
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


def _write_out_of_order_sync_primary_mcap(path: Path) -> None:
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


def _write_mixed_state_shape_mcap(path: Path) -> None:
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
        messages = [
            (state_channel, 0, b'{"q":[0]}', 0, 1),
            (state_channel, 500_000_000, b'{"diagnostic":"ok"}', 500_000_000, 2),
            (state_channel, 1_000_000_000, b'{"q":[1]}', 1_000_000_000, 3),
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
    frames = row["records"]
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


def test_mcap_reader_maps_fields_and_aligns_to_sync_primary(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(
        str(path),
        fields={
            "state": "/joint_states.q",
            "target": "/cmd.target",
        },
        sync_primary="state",
    ).materialize()[0]

    frames = row["records"]
    assert frames.table.num_rows == 3
    assert frames.column("timestamp").to_pylist() == [0.0, 1.0, 2.0]
    assert frames.column("state").to_pylist() == [[1, 2], [3, 4], [5, 6]]
    assert frames.column("target").to_pylist() == [[10], [20], [20]]
    assert "mcap.state.skew_ms" not in frames.table.column_names
    assert frames.column("mcap.target.skew_ms").to_pylist() == [100.0, -100.0, -1100.0]
    assert row["fps"] == 1.0


def test_mcap_reader_interpolates_numeric_fields(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(
        str(path),
        fields={"state": "/joint_states.q", "target": "/cmd.target"},
        sync_primary="target",
        sync_method="interpolate",
        include_skew=False,
    ).materialize()[0]

    assert row["records"].column("timestamp").to_pylist() == [0.1, 0.9]
    assert row["records"].column("state").to_pylist() == [
        pytest.approx([1.2, 2.2]),
        pytest.approx([2.8, 3.8]),
    ]


def test_mcap_reader_holds_previous_field_value(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(
        str(path),
        fields={"state": "/joint_states.q", "target": "/cmd.target"},
        sync_primary="target",
        sync_method="hold",
        include_skew=False,
    ).materialize()[0]

    assert row["records"].column("timestamp").to_pylist() == [0.1, 0.9]
    assert row["records"].column("state").to_pylist() == [[1, 2], [1, 2]]


def test_mcap_reader_hold_does_not_use_future_values(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(
        str(path),
        fields={"state": "/joint_states.q", "target": "/cmd.target"},
        sync_primary="state",
        sync_method="hold",
        include_skew=False,
    ).materialize()[0]

    assert row["records"].column("target").to_pylist() == [None, [20], [20]]


def test_mcap_reader_can_align_to_unselected_sync_primary_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(
        str(path),
        fields={"target": "/cmd.target"},
        sync_primary="/joint_states.q",
        include_skew=False,
    ).materialize()[0]

    frames = row["records"]
    assert frames.table.column_names == ["frame_index", "timestamp", "target"]
    assert frames.table.num_rows == 3
    assert frames.column("timestamp").to_pylist() == [0.0, 1.0, 2.0]
    assert frames.column("target").to_pylist() == [[10], [20], [20]]


def test_mcap_reader_sorts_sync_primary_events_before_alignment(tmp_path: Path) -> None:
    path = tmp_path / "out-of-order.mcap"
    _write_out_of_order_sync_primary_mcap(path)

    row = read_mcap(
        str(path),
        fields={"state": "/state.q", "target": "/target.u"},
        sync_primary="state",
        include_skew=False,
    ).materialize()[0]

    frames = row["records"]
    assert frames.column("timestamp").to_pylist() == [0.0, 1.0, 2.0]
    assert frames.column("state").to_pylist() == [[0], [1], [2]]
    assert frames.column("target").to_pylist() == [[0], [1], [2]]
    assert row["fps"] == 1.0


def test_mcap_reader_filters_sync_primary_subfield_events(tmp_path: Path) -> None:
    path = tmp_path / "mixed-state-shape.mcap"
    _write_mixed_state_shape_mcap(path)

    row = read_mcap(
        str(path),
        fields={"state": "/state.q"},
        sync_primary="/state.q",
    ).materialize()[0]

    frames = row["records"]
    assert frames.column("timestamp").to_pylist() == [0.0, 1.0]
    assert frames.column("state").to_pylist() == [[0], [1]]
    assert row["fps"] == 1.0


def test_mcap_reader_aligns_same_topic_non_primary_subfields(tmp_path: Path) -> None:
    path = tmp_path / "mixed-state-shape.mcap"
    _write_mixed_state_shape_mcap(path)

    row = read_mcap(
        str(path),
        fields={"state": "/state.q", "diagnostic": "/state.diagnostic"},
        sync_primary="state",
        include_skew=False,
    ).materialize()[0]

    frames = row["records"]
    assert frames.column("timestamp").to_pylist() == [0.0, 1.0]
    assert frames.column("state").to_pylist() == [[0], [1]]
    assert frames.column("diagnostic").to_pylist() == ["ok", "ok"]


def test_mcap_reader_builds_video_frame_arrays_for_robot_rows(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = (
        read_mcap(
            str(path),
            fields={"state": "/joint_states.q"},
            videos={"front": "/image.frame"},
            sync_primary="state",
            fps=12,
        )
        .to_robot_rows(
            nested_frames_key="records",
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


def test_mcap_reader_decodes_h264_video_messages(tmp_path: Path) -> None:
    path = tmp_path / "h264.mcap"
    _write_h264_mcap(path)

    row = read_mcap(
        str(path),
        fields={"state": "/state.q"},
        videos={"front": "/video"},
        sync_primary="front",
        fps=2,
    ).materialize()[0]

    assert row["records"].column("timestamp").to_pylist() == [0.0, 0.5]
    assert row["records"].column("state").to_pylist() == [[1], [2]]
    video = row["videos"]["front"]
    assert video.frame_count == 2
    red, green = [frame[0, 0] for frame in video.iter_frame_arrays()]
    assert red[0] > 200 and red[1] < 20 and red[2] < 20
    assert green[0] < 20 and green[1] > 200 and green[2] < 20


def test_mcap_reader_omits_video_topics_from_default_fields(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(str(path), videos={"front": "/image.frame"}).materialize()[0]

    assert "/image.frame" not in row["records"].table.column_names
    assert row["videos"]["front"].frame_count == 2


def test_mcap_reader_reads_non_seekable_streams(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(
        (str(path), _NonSeekableLocalFileSystem()),
        fields={"state": "/joint_states.q"},
    ).materialize()[0]

    assert row["records"].column("state").to_pylist() == [[1, 2], [3, 4], [5, 6]]


def test_mcap_reader_stream_episodes_falls_back_for_non_seekable_streams(
    tmp_path: Path,
) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    with pytest.warns(RuntimeWarning, match="fell back to buffered reading"):
        rows = read_mcap(
            (str(path), _NonSeekableLocalFileSystem()),
            fields={"state": "/joint_states.q"},
            sync_primary="state",
            episode_splitting={"time_gap_s": 0.5},
            stream_episodes=True,
        ).materialize()

    assert [row["records"].column("state").to_pylist()[0] for row in rows] == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]


def test_mcap_reader_treats_string_fields_as_one_source(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(str(path), fields="/cmd.target").materialize()[0]

    assert row["records"].table.column_names == [
        "frame_index",
        "timestamp",
        "/cmd.target",
    ]
    assert row["records"].column("/cmd.target").to_pylist() == [[10], [20]]


def test_mcap_reader_keeps_explicit_subfield_decoding_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Decoded:
        def __init__(self, value: int):
            self.value = value

    class DecoderFactory:
        def decoder_for(self, message_encoding: str, schema: Any):
            if message_encoding == "custom":
                return lambda data: Decoded(int(data))
            return None

    path = tmp_path / "custom.mcap"
    _write_custom_encoding_mcap(path)
    monkeypatch.setattr(mcap_reader, "_decoder_factories", lambda: [DecoderFactory()])
    monkeypatch.setattr(
        mcap_reader,
        "_plain_value",
        lambda value: pytest.fail("explicit subfields should not normalize messages"),
    )

    row = read_mcap(str(path), fields={"value": "/custom.value"}).materialize()[0]

    assert row["records"].column("value").to_pylist() == [7]


def test_mcap_reader_preserves_explicit_empty_fields(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(str(path), fields={}).materialize()[0]

    assert row["records"].table.column_names == []


def test_mcap_reader_rejects_unknown_field_source(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    with pytest.raises(KeyError, match="/missing.value"):
        read_mcap(str(path), fields={"bad": "/missing.value"}).materialize()


def test_mcap_reader_rejects_file_path_column_collisions(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    with pytest.raises(ValueError, match="file_path_column"):
        read_mcap(str(path), file_path_column="records").materialize()


def test_mcap_reader_rejects_reserved_frame_field_names(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    with pytest.raises(ValueError, match="reserved frame columns"):
        read_mcap(str(path), fields={"timestamp": "/cmd.target"}).materialize()


def test_mcap_reader_splits_on_time_gaps(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    rows = read_mcap(
        str(path),
        fields={"state": "/joint_states.q"},
        sync_primary="state",
        episode_splitting={"time_gap_s": 0.5},
    ).materialize()

    assert [row["episode_index"] for row in rows] == [0, 1, 2]
    assert [row["records"].table.num_rows for row in rows] == [1, 1, 1]
    assert [row["records"].column("state").to_pylist()[0] for row in rows] == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]


def test_mcap_reader_streams_time_gap_episodes(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    rows = read_mcap(
        str(path),
        fields={"state": "/joint_states.q"},
        sync_primary="state",
        episode_splitting={"time_gap_s": 0.5},
        stream_episodes=True,
    ).materialize()

    assert [row["episode_index"] for row in rows] == [0, 1, 2]
    assert [row["records"].column("state").to_pylist()[0] for row in rows] == [
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
        sync_primary="state",
        episode_splitting={"marker_topic": "/episode_start"},
    ).materialize()

    assert [row["episode_index"] for row in rows] == [0, 1]
    assert [row["records"].column("state").to_pylist() for row in rows] == [
        [[1], [2]],
        [[3]],
    ]


def test_mcap_reader_streams_marker_episodes(tmp_path: Path) -> None:
    path = tmp_path / "markers.mcap"
    _write_marker_mcap(path)

    rows = read_mcap(
        str(path),
        fields={"state": "/state.q"},
        sync_primary="state",
        episode_splitting={"marker_topic": "/episode_start"},
        stream_episodes=True,
    ).materialize()

    assert [row["records"].column("state").to_pylist() for row in rows] == [
        [[1], [2]],
        [[3]],
    ]


def test_mcap_reader_preserves_selected_empty_episode_fields(tmp_path: Path) -> None:
    path = tmp_path / "markers-sparse-action.mcap"
    _write_marker_mcap_with_sparse_action(path)

    rows = read_mcap(
        str(path),
        fields={"state": "/state.q", "action": "/action.u"},
        sync_primary="state",
        episode_splitting={"marker_topic": "/episode_start"},
        include_skew=False,
    ).materialize()

    assert [row["records"].column("action").to_pylist() for row in rows] == [
        [[10]],
        [None],
    ]


def test_mcap_reader_sorts_marker_events_before_splitting(tmp_path: Path) -> None:
    path = tmp_path / "out-of-order-markers.mcap"
    _write_out_of_order_marker_mcap(path)

    rows = read_mcap(
        str(path),
        fields={"state": "/state.q"},
        sync_primary="state",
        episode_splitting={"marker_topic": "/episode_start"},
    ).materialize()

    assert [row["records"].column("state").to_pylist() for row in rows] == [
        [[1], [2]],
        [[3]],
    ]


def test_mcap_reader_reads_marker_topic_with_selected_fields(tmp_path: Path) -> None:
    path = tmp_path / "markers.mcap"
    _write_marker_mcap(path)

    rows = read_mcap(
        str(path),
        fields={"state": "/state.q"},
        sync_primary="state",
        episode_splitting={"marker_topic": "/episode_start"},
    ).materialize()

    assert [row["records"].column("state").to_pylist() for row in rows] == [
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

    assert rows[0]["records"].table.column_names == [
        "frame_index",
        "timestamp",
        "/state.q",
    ]


def test_mcap_reader_default_fields_union_optional_keys(tmp_path: Path) -> None:
    path = tmp_path / "sparse-edge.mcap"
    _write_sparse_edge_mcap(path)

    row = read_mcap(str(path)).materialize()[0]

    frames = row["records"]
    assert "/event.a" in frames.table.column_names
    assert "/event.b" in frames.table.column_names
    assert frames.column("/event.a").to_pylist() == [1, None, None, None]
    assert frames.column("/event.b").to_pylist() == [None, 2, None, None]


def test_mcap_reader_sparse_mode_preserves_duplicate_timestamps(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sparse-edge.mcap"
    _write_sparse_edge_mcap(path)

    row = read_mcap(str(path), fields="/state.q").materialize()[0]

    frames = row["records"]
    assert frames.column("timestamp").to_pylist() == [1e-08, 1e-08]
    assert frames.column("/state.q").to_pylist() == [[1], [2]]


def test_mcap_reader_aligned_mode_preserves_duplicate_sync_primary_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sparse-edge.mcap"
    _write_sparse_edge_mcap(path)

    row = read_mcap(
        str(path),
        fields={"state": "/state.q"},
        sync_primary="state",
        include_skew=False,
    ).materialize()[0]

    frames = row["records"]
    assert frames.column("timestamp").to_pylist() == [1e-08, 1e-08]
    assert frames.column("state").to_pylist() == [[1], [2]]


def test_mcap_reader_preserves_duplicate_subfields_for_topic_sync_primary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sparse-edge.mcap"
    _write_sparse_edge_mcap(path)

    row = read_mcap(
        str(path),
        fields={"state": "/state.q"},
        sync_primary="/state",
        include_skew=False,
    ).materialize()[0]

    frames = row["records"]
    assert frames.column("timestamp").to_pylist() == [1e-08, 1e-08]
    assert frames.column("state").to_pylist() == [[1], [2]]


def test_mcap_reader_rejects_generated_skew_field_name_collisions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sparse-edge.mcap"
    _write_sparse_edge_mcap(path)

    with pytest.raises(ValueError, match="generated skew/timestamp"):
        read_mcap(
            str(path),
            fields={
                "clock": "/event.a",
                "target": "/state.q",
                "mcap.target.skew_ms": "/event.b",
            },
            sync_primary="clock",
        ).materialize()


def test_mcap_reader_sorts_unaligned_video_frames(tmp_path: Path) -> None:
    path = tmp_path / "out-of-order-video.mcap"
    _write_out_of_order_video_mcap(path)

    row = read_mcap(str(path), videos={"front": "/image.frame"}).materialize()[0]

    assert [
        frame[0, 0].tolist() for frame in row["videos"]["front"].iter_frame_arrays()
    ] == [
        [1, 2, 3],
        [4, 5, 6],
    ]


def test_mcap_reader_preserves_duplicate_sync_primary_video_frames(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate-video-sync-primary.mcap"
    _write_duplicate_video_timestamp_mcap(path)

    row = read_mcap(
        str(path),
        videos={"front": "/image.frame"},
        sync_primary="front",
        fps=1,
    ).materialize()[0]

    assert row["records"].table.num_rows == 2
    assert [
        frame[0, 0].tolist() for frame in row["videos"]["front"].iter_frame_arrays()
    ] == [
        [1, 2, 3],
        [4, 5, 6],
    ]


def test_mcap_reader_rejects_missing_hold_aligned_video_frame(
    tmp_path: Path,
) -> None:
    path = tmp_path / "late-video.mcap"
    _write_late_video_mcap(path)

    with pytest.raises(ValueError, match="no aligned frame"):
        read_mcap(
            str(path),
            fields={"state": "/state.q"},
            videos={"front": "/image.frame"},
            sync_primary="state",
            sync_method="hold",
            fps=1,
        ).materialize()


def test_mcap_reader_keeps_videos_aligned_to_empty_sync_primary_episode(
    tmp_path: Path,
) -> None:
    path = tmp_path / "video-only-episode.mcap"
    _write_marker_mcap_with_video_only_episode(path)

    rows = read_mcap(
        str(path),
        fields={"state": "/state.q"},
        videos={"front": "/image.frame"},
        sync_primary="state",
        episode_splitting={"marker_topic": "/episode_start"},
        fps=1,
    ).materialize()

    assert rows[0]["records"].table.num_rows == 1
    assert rows[0]["videos"]["front"].frame_count == 1
    assert rows[1]["records"].table.num_rows == 0
    assert "videos" not in rows[1]


def test_mcap_reader_default_fields_resolve_sync_primary_from_file_topics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "video-only-episode.mcap"
    _write_marker_mcap_with_video_only_episode(path)

    rows = read_mcap(
        str(path),
        videos={"front": "/image.frame"},
        sync_primary="/state.q",
        episode_splitting={"marker_topic": "/episode_start"},
        fps=1,
    ).materialize()

    assert rows[0]["records"].column("/state.q").to_pylist() == [[1]]
    assert rows[0]["videos"]["front"].frame_count == 1
    assert rows[1]["records"].table.num_rows == 0
    assert "videos" not in rows[1]


def test_mcap_reader_preserves_fractional_video_fps(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    row = read_mcap(
        str(path),
        videos={"front": "/image.frame"},
        fps=29.97,
    ).materialize()[0]

    assert row["fps"] == 29.97
    assert row["videos"]["front"].fps == 29.97


def test_mcap_reader_rounds_near_integer_inferred_video_fps(tmp_path: Path) -> None:
    path = tmp_path / "rounded-fps.mcap"
    _write_nanosecond_rounded_fps_mcap(path)

    row = read_mcap(
        str(path),
        fields={"state": "/state.q"},
        videos={"front": "/image.frame"},
        sync_primary="state",
    ).materialize()[0]

    assert row["fps"] == 30.0
    assert row["videos"]["front"].fps == 30
    assert row["videos"]["front"].frame_count == 3


def test_mcap_reader_decodes_compressed_image_byte_lists(tmp_path: Path) -> None:
    from PIL import Image

    image = Image.new("RGB", (1, 1), (1, 2, 3))
    out = tmp_path / "frame.jpg"
    image.save(out)

    frame = _frame_from_value({"format": "jpeg", "data": list(out.read_bytes())})

    assert frame.shape == (1, 1, 3)


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


def test_mcap_reader_decodes_sixteen_bit_ros_images() -> None:
    rgb = _ros_image_frame(
        {
            "height": 1,
            "width": 1,
            "encoding": "rgb16",
            "step": 8,
            "data": np.array([257, 514, 771, 999], dtype="<u2").tobytes(),
        }
    )
    bgr = _ros_image_frame(
        {
            "height": 1,
            "width": 1,
            "encoding": "bgr16",
            "step": 6,
            "data": np.array([257, 514, 771], dtype="<u2").tobytes(),
        }
    )

    assert rgb.tolist() == [[[1, 2, 3]]]
    assert bgr.tolist() == [[[3, 2, 1]]]


def test_mcap_reader_rejects_unknown_episode_splitting(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    with pytest.raises(ValueError, match="episode_splitting must be"):
        read_mcap(str(path), episode_splitting="marker")


def test_mcap_reader_rejects_unknown_sync_method(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    with pytest.raises(ValueError, match="sync_method must be"):
        read_mcap(str(path), sync_method=cast(Any, "linear")).materialize()


def test_mcap_reader_plans_files_atomically(tmp_path: Path) -> None:
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    _write_mcap(first)
    _write_mcap(second)

    reader = McapReader([str(first), str(second)], target_shard_bytes=1)

    assert len(reader.list_shards()) == 2
