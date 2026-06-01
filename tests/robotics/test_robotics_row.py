from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from refiner.pipeline import from_items
from refiner.pipeline.data import datatype
from refiner.pipeline.expressions import col
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.tabular import Tabular
from refiner.robotics import motion_trim
from refiner.robotics.row import (
    RoboticsRow,
    _robot_row_converter,
)
from refiner.video import VideoBytes, VideoFile, VideoFrameArray


def _robot_row(row: Row, **kwargs: Any) -> RoboticsRow:
    return cast(RoboticsRow, _robot_row_converter(**kwargs)(row))


def _robot_rows(
    rows: Row | Tabular,
    *,
    layout: str = "episode_rows",
    **kwargs: Any,
) -> list[RoboticsRow]:
    assert layout == "episode_rows"
    if isinstance(rows, Tabular):
        converter = _robot_row_converter(**cast(Any, {**kwargs, "schema": rows.schema}))
        return [cast(RoboticsRow, converter(row)) for row in rows]
    converter = _robot_row_converter(**kwargs)
    return [cast(RoboticsRow, converter(rows))]


def test_to_robot_rows_does_not_treat_video_uri_frames_as_frame_table() -> None:
    row = DictRow(
        {
            "id": "episode-1",
            "task": "pick the cup",
            "frames": "clips/episode-1.mp4",
        }
    )

    robotics_row = _robot_row(
        row,
        episode_id_key="id",
        task_key="task",
        video_keys={"observation.images.main": "frames"},
    )

    assert robotics_row.episode_id == "episode-1"
    assert robotics_row.task == "pick the cup"
    assert robotics_row.num_frames == -1
    video = robotics_row.videos["observation.images.main"]
    assert isinstance(video, VideoFile)
    assert video.uri.endswith("/clips/episode-1.mp4")


def test_to_robot_rows_exposes_stats_and_embedded_video_bytes() -> None:
    row = DictRow(
        {
            "episode_id": "episode-1",
            "camera_bytes": b"video bytes",
            "stats/action/min": [0.0],
            "stats/action/max": [1.0],
        }
    )

    robotics_row = _robot_row(
        row,
        video_keys={"observation.images.main": "camera_bytes"},
    )

    video = robotics_row.videos["observation.images.main"]
    assert isinstance(video, VideoBytes)
    assert video.open().read() == b"video bytes"
    assert robotics_row.stats["action"]["min"] == [0.0]
    assert robotics_row.stats["action"]["max"] == [1.0]

    without_action_stats = robotics_row.drop_stats("action")
    assert without_action_stats.stats == {}


def test_to_robot_rows_uses_video_asset_schema() -> None:
    table = Tabular.from_rows(
        [
            DictRow({"episode_id": "episode-1", "camera": "clips/episode-1.mp4"}),
        ],
        schema=datatype.schema_with_dtypes(
            None,
            {"camera": datatype.video_path()},
        ),
    )

    robotics_row = list(_robot_rows(table, episode_id_key="episode_id"))[0]

    video = robotics_row.videos["camera"]
    assert isinstance(video, VideoFile)
    assert video.uri.endswith("/clips/episode-1.mp4")


def test_to_robot_rows_uses_video_frame_array_asset_schema() -> None:
    frames = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    table = Tabular.from_rows(
        [
            DictRow({"episode_id": "episode-1", "camera": frames}),
        ],
        schema=datatype.schema_with_dtypes(
            None,
            {"camera": datatype.video_frame_array()},
        ),
    )

    robotics_row = list(_robot_rows(table, episode_id_key="episode_id", fps=12))[0]

    video = robotics_row.videos["camera"]
    assert isinstance(video, VideoFrameArray)
    video_frames = list(video.iter_frame_arrays())
    assert len(video_frames) == 2
    assert video_frames[0].shape == (4, 5, 3)
    assert video.fps == 12


def test_to_robot_rows_accepts_unmapped_key_iterables() -> None:
    row = DictRow(
        {
            "episode_id": "episode-1",
            "action": [[0.0]],
            "observation.state": [[1.0]],
            "qvel": [[2.0]],
            "front_video": "clips/episode-1.mp4",
        }
    )

    robotics_row = _robot_row(
        row,
        extra_observation_keys=("qvel",),
        video_keys=("front_video",),
    )

    assert robotics_row.observations("qvel") == [[2.0]]
    video = robotics_row.videos["front_video"]
    assert isinstance(video, VideoFile)
    assert video.uri.endswith("/clips/episode-1.mp4")


def test_to_robot_rows_iteration_hides_nested_source_containers() -> None:
    row = DictRow(
        {
            "episode_id": "episode-1",
            "label": "keep",
            "obs": {
                "qpos": [[1.0], [2.0]],
                "front": "clips/episode-1.mp4",
            },
        }
    )

    robotics_row = _robot_row(
        row,
        episode_id_key="episode_id",
        timestamp_key=None,
        action_key=None,
        state_key="obs/qpos",
        video_keys={"observation.images.front": "obs/front"},
    )

    assert dict(cast(Row, robotics_row).items()) == {
        "episode_id": "episode-1",
        "label": "keep",
    }
    assert robotics_row.states == [[1.0], [2.0]]
    video = robotics_row.videos["observation.images.front"]
    assert isinstance(video, VideoFile)
    assert video.uri.endswith("/clips/episode-1.mp4")


def test_to_robot_rows_accepts_tabular_input() -> None:
    table = Tabular.from_rows(
        [
            DictRow({"episode_id": "episode-1", "task": "pick"}),
            DictRow({"episode_id": "episode-2", "task": "place"}),
        ]
    )

    rows = list(_robot_rows(table, episode_id_key="episode_id", task_key="task"))

    assert [row.episode_id for row in rows] == ["episode-1", "episode-2"]
    assert [row.task for row in rows] == ["pick", "place"]


def test_to_robot_rows_defaults_do_not_assume_fps_or_robot_type_keys() -> None:
    row = DictRow(
        {
            "episode_id": "episode-1",
            "fps": 10,
            "robot_type": "mockbot",
        }
    )

    robotics_row = _robot_row(row)

    assert robotics_row.fps is None
    assert robotics_row.robot_type is None


def test_to_robot_rows_defaults_to_no_episode_id_source() -> None:
    row = DictRow({"episode_id": "source-value", "action": [[0.0]]})

    robotics_row = _robot_row(row, timestamp_key=None, state_key=None)

    assert robotics_row.episode_id == "-1"
    assert dict(cast(Row, robotics_row).items())["episode_id"] == "source-value"


def test_to_robot_rows_accepts_literal_fps_and_robot_type() -> None:
    row = DictRow({"episode_id": "episode-1"})

    robotics_row = _robot_row(row, fps=20.0, robot_type="koch")

    assert robotics_row.fps == 20.0
    assert robotics_row.robot_type == "koch"


def test_to_robot_rows_infers_timestamps_from_literal_fps() -> None:
    row = DictRow({"episode_id": "episode-1", "action": [[0.0], [0.1], [0.2]]})

    robotics_row = _robot_row(row, fps=10.0, state_key=None)

    assert robotics_row.to_frame_table().column(
        "timestamp"
    ).to_pylist() == pytest.approx(
        [
            0.0,
            0.1,
            0.2,
        ]
    )


def test_to_robot_rows_preserves_explicit_fps_and_robot_type_keys() -> None:
    row = DictRow(
        {
            "episode_id": "episode-1",
            "source_fps": "15",
            "source_robot": "aloha",
        }
    )

    robotics_row = _robot_row(
        row,
        fps_key="source_fps",
        robot_type_key="source_robot",
    )

    assert robotics_row.fps == 15.0
    assert robotics_row.robot_type == "aloha"


def test_pipeline_to_robot_rows_forwards_literals_and_explicit_keys() -> None:
    literal_row = (
        from_items([{"episode_id": "episode-1"}])
        .to_robot_rows(
            fps=12.5,
            robot_type="literalbot",
        )
        .materialize()[0]
    )
    keyed_row = (
        from_items(
            [{"episode_id": "episode-2", "source_fps": 7, "source_robot": "keybot"}]
        )
        .to_robot_rows(
            fps_key="source_fps",
            robot_type_key="source_robot",
        )
        .materialize()[0]
    )

    assert isinstance(literal_row, RoboticsRow)
    assert literal_row.fps == 12.5
    assert literal_row.robot_type == "literalbot"
    assert isinstance(keyed_row, RoboticsRow)
    assert keyed_row.fps == 7.0
    assert keyed_row.robot_type == "keybot"


def test_to_robot_rows_preserves_robotics_view_after_vectorized_filter() -> None:
    rows = (
        from_items(
            [
                {
                    "episode_id": "episode-0",
                    "keep": False,
                    "frames": [
                        {
                            "timestamp": 0.0,
                            "action": [0.0],
                            "observation.state": [1.0],
                        }
                    ],
                },
                {
                    "episode_id": "episode-1",
                    "keep": True,
                    "frames": [
                        {
                            "timestamp": 0.0,
                            "action": [1.0],
                            "observation.state": [2.0],
                        }
                    ],
                },
            ]
        )
        .to_robot_rows(
            episode_id_key="episode_id",
            nested_frames_key="frames",
        )
        .filter(col("keep"))
        .materialize()
    )

    assert len(rows) == 1
    row = cast(RoboticsRow, rows[0])
    assert isinstance(row, RoboticsRow)
    assert row.episode_id == "episode-1"
    assert row.num_frames == 1
    assert row.actions.to_pylist() == [[1.0]]
    assert row.states.to_pylist() == [[2.0]]


def test_to_robot_rows_preserves_nested_frame_side_data_after_mutation() -> None:
    rows = (
        from_items(
            [
                {
                    "episode_id": "episode-1",
                    "keep": True,
                    "frames": [
                        {
                            "timestamp": 0.0,
                            "action": [0.0],
                            "observation.state": [1.0],
                        }
                    ],
                }
            ]
        )
        .to_robot_rows(
            episode_id_key="episode_id",
            nested_frames_key="frames",
        )
        .map(lambda row: cast(RoboticsRow, row).with_actions([[2.0]]))
        .filter(col("keep"))
        .materialize()
    )

    assert len(rows) == 1
    row = cast(RoboticsRow, rows[0])
    assert isinstance(row, RoboticsRow)
    assert row.actions.to_pylist() == [[2.0]]
    assert row.states.to_pylist() == [[1.0]]


def test_to_robot_rows_realigns_side_data_when_arrow_rows_are_reordered() -> None:
    table = (
        from_items(
            [
                {
                    "episode_id": "episode-0",
                    "rank": 1,
                    "frames": [
                        {
                            "timestamp": 0.0,
                            "action": [0.0],
                            "observation.state": [10.0],
                        }
                    ],
                },
                {
                    "episode_id": "episode-1",
                    "rank": 0,
                    "frames": [
                        {
                            "timestamp": 0.0,
                            "action": [1.0],
                            "observation.state": [20.0],
                        }
                    ],
                },
            ]
        )
        .to_robot_rows(
            episode_id_key="episode_id",
            nested_frames_key="frames",
        )
        .map(lambda row: cast(RoboticsRow, row).with_actions([[row["rank"]]]))
        .map_table(lambda table: table.sort_by([("rank", "ascending")]))
        .materialize()
    )

    rows = [cast(RoboticsRow, row) for row in table]
    assert [row.episode_id for row in rows] == ["episode-1", "episode-0"]
    assert [row.actions.to_pylist() for row in rows] == [[[0]], [[1]]]
    assert [row.states.to_pylist() for row in rows] == [[[20.0]], [[10.0]]]


def test_to_robot_rows_falls_back_for_mixed_row_batches() -> None:
    rows = (
        from_items(
            [
                {
                    "episode_id": "episode-1",
                    "keep": True,
                    "action": [[0.0]],
                    "observation.state": [[1.0]],
                },
                {
                    "episode_id": "episode-2",
                    "keep": True,
                    "action": [[1.0]],
                    "observation.state": [[2.0]],
                },
            ]
        )
        .to_robot_rows(episode_id_key="episode_id")
        .map(
            lambda row: (
                row
                if cast(RoboticsRow, row).episode_id == "episode-1"
                else DictRow({"episode_id": row["episode_id"], "keep": row["keep"]})
            )
        )
        .filter(col("keep"))
        .materialize()
    )

    assert [row["episode_id"] for row in rows] == ["episode-1", "episode-2"]


def test_to_robot_rows_output_can_be_motion_trimmed() -> None:
    row = DictRow(
        {
            "episode_id": "episode-1",
            "timestamp": [0.0, 0.1, 0.2, 0.3],
            "action": [[0.0], [0.0], [1.0], [1.0]],
            "observation.state": [[0.0], [0.0], [0.0], [0.0]],
        }
    )
    robotics_row = _robot_row(row)

    trimmed = cast(RoboticsRow, motion_trim(threshold=0.25)(cast(Row, robotics_row)))

    assert trimmed.num_frames == 2
    assert trimmed.timestamps == pytest.approx([0.0, 0.1])


def test_to_robot_rows_maps_alternate_semantic_keys() -> None:
    row = DictRow(
        {
            "episode_id": "episode-1",
            "time_s": [0.0, 0.1, 0.2],
            "actions": [[0.0], [1.0], [1.0]],
            "obs/qpos": [[0.0], [0.0], [0.0]],
            "camera": ["frame-0", "frame-1", "frame-2"],
        }
    )

    robotics_row = _robot_row(
        row,
        timestamp_key="time_s",
        action_key="actions",
        state_key="obs/qpos",
        extra_observation_keys={"images.main": "camera"},
    )

    assert robotics_row.num_frames == 3
    assert robotics_row.actions == [[0.0], [1.0], [1.0]]
    assert robotics_row.observations("state") == [[0.0], [0.0], [0.0]]
    assert set(robotics_row.observations()) == {"state", "images.main"}
    assert robotics_row.observations("images.main") == [
        "frame-0",
        "frame-1",
        "frame-2",
    ]
    assert robotics_row.to_frame_table().table.column_names == [
        "timestamp",
        "action",
        "observation.state",
        "observation.images.main",
    ]
    updated = robotics_row.with_actions([[2.0], [2.0], [2.0]]).with_observation(
        "state",
        [[3.0], [3.0], [3.0]],
    )
    assert updated.actions == [[2.0], [2.0], [2.0]]
    assert updated.states == [[3.0], [3.0], [3.0]]


def test_to_robot_rows_concatenates_tuple_state_key_sources() -> None:
    row = DictRow(
        {
            "episode_id": "episode-1",
            "timestamp": [0.0, 0.1, 0.2],
            "actions": [[0.0], [0.1], [0.2]],
            "obs": {
                "joint_pos": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                "joint_vel": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            },
            "eef_pos": np.asarray(
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                dtype=np.float32,
            ),
        }
    )

    robotics_row = _robot_row(
        row,
        action_key="actions",
        state_key=("obs/joint_pos", "obs/joint_vel", "eef_pos"),
    )

    expected_state = [
        [1.0, 2.0, 0.1, 0.2, 10.0, 11.0, 12.0],
        [3.0, 4.0, 0.3, 0.4, 13.0, 14.0, 15.0],
        [5.0, 6.0, 0.5, 0.6, 16.0, 17.0, 18.0],
    ]
    assert robotics_row.states == expected_state
    assert robotics_row.observations("state") == expected_state

    frame_table = robotics_row.to_frame_table()
    assert frame_table.table.column_names == [
        "timestamp",
        "action",
        "observation.state",
    ]
    assert frame_table.column("observation.state").to_pylist() == expected_state

    selected = robotics_row.select_frames([2, 0])
    assert selected.states == [expected_state[2], expected_state[0]]
    assert selected.actions == [[0.2], [0.0]]


def test_to_robot_rows_concatenates_tuple_state_key_nested_frames() -> None:
    row = DictRow(
        {
            "episode_id": "episode-1",
            "frames": Tabular.from_rows(
                [
                    DictRow(
                        {
                            "actions": [0.0],
                            "obs/joint_pos": [1.0, 2.0],
                            "obs/joint_vel": [0.1, 0.2],
                            "eef_pos": [10.0, 11.0, 12.0],
                        }
                    ),
                    DictRow(
                        {
                            "actions": [1.0],
                            "obs/joint_pos": [3.0, 4.0],
                            "obs/joint_vel": [0.3, 0.4],
                            "eef_pos": [13.0, 14.0, 15.0],
                        }
                    ),
                ]
            ),
        }
    )

    robotics_row = _robot_row(
        row,
        nested_frames_key="frames",
        timestamp_key=None,
        action_key="actions",
        state_key=("obs/joint_pos", "obs/joint_vel", "eef_pos"),
    )

    expected_state = [
        [1.0, 2.0, 0.1, 0.2, 10.0, 11.0, 12.0],
        [3.0, 4.0, 0.3, 0.4, 13.0, 14.0, 15.0],
    ]
    assert robotics_row.states.to_pylist() == expected_state
    assert robotics_row.observations("state").to_pylist() == expected_state
    assert (
        robotics_row.to_frame_table().column("observation.state").to_pylist()
        == expected_state
    )


def test_to_robot_rows_maps_nested_semantic_keys() -> None:
    row = DictRow(
        {
            "episode_id": "episode-1",
            "frames": Tabular.from_rows(
                [
                    DictRow({"time_s": 0.0, "actions": [0.0]}),
                    DictRow({"time_s": 0.1, "actions": [1.0]}),
                ]
            ),
        }
    )

    robotics_row = _robot_row(
        row,
        nested_frames_key="frames",
        timestamp_key="time_s",
        action_key="actions",
    )

    assert robotics_row.num_frames == 2
    assert robotics_row.timestamps.to_pylist() == [0.0, 0.1]
    assert robotics_row.to_frame_table().table.column_names == ["timestamp", "action"]


def test_to_robot_rows_flattens_nested_frame_dicts() -> None:
    row = DictRow(
        {
            "episode_id": "episode-1",
            "steps": [
                {"observation": {"state": [0.0]}, "action": [0.0]},
                {"observation": {"state": [1.0]}, "action": [1.0]},
            ],
        }
    )

    robotics_row = _robot_row(
        row,
        nested_frames_key="steps",
        state_key="observation/state",
        action_key="action",
        timestamp_key=None,
    )

    assert robotics_row.num_frames == 2
    assert robotics_row.states.to_pylist() == [[0.0], [1.0]]


def test_motion_trim_works_with_mapped_semantic_keys() -> None:
    row = DictRow(
        {
            "episode_id": "episode-1",
            "time_s": [0.0, 0.1, 0.2, 0.3],
            "actions": [[0.0], [0.0], [1.0], [1.0]],
            "qpos": [[0.0], [0.0], [0.0], [0.0]],
        }
    )
    robotics_row = _robot_row(
        row,
        timestamp_key="time_s",
        action_key="actions",
        state_key="qpos",
    )

    trimmed = cast(RoboticsRow, motion_trim(threshold=0.25)(cast(Row, robotics_row)))

    assert trimmed.num_frames == 2
    assert trimmed.timestamps == pytest.approx([0.0, 0.1])
    assert "time_s" not in trimmed.to_frame_table().table.column_names
    assert "timestamp" in trimmed.to_frame_table().table.column_names
