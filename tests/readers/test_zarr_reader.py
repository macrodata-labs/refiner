from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import zarr

import refiner as mdr
from refiner.robotics.row import RoboticsRow
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.shard import RowRangeDescriptor


def _write_policy_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset(
        "data/action",
        data=np.asarray([[0.0], [0.1], [1.0], [1.1], [1.2]], dtype=np.float32),
    )
    root.create_dataset(
        "data/state",
        data=np.asarray([[10.0], [10.1], [20.0], [20.1], [20.2]], dtype=np.float32),
    )
    root.create_dataset(
        "data/rgb",
        data=np.arange(5 * 4 * 4 * 3, dtype=np.uint8).reshape(5, 4, 4, 3),
    )
    root.create_dataset("meta/episode_ends", data=np.asarray([2, 5], dtype=np.int64))
    root.attrs["dataset_id"] = "pusht"
    root.attrs["task"] = "push tee"


def test_read_zarr_rejects_reserved_file_path_output_name(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)

    with pytest.raises(ValueError, match="reserved output names"):
        mdr.read_zarr(path, arrays={"file_path": "data/action"})


def test_read_zarr_reads_selected_arrays_and_attrs(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)

    row = mdr.read_zarr(
        path,
        arrays={
            "action": "data/action",
            "state": "data/state",
            "episode_ends": "meta/episode_ends",
        },
        attrs={"task": "task"},
        file_path_column=None,
    ).take(1)[0]

    assert row["task"] == "push tee"
    assert row["episode_ends"].tolist() == [2, 5]
    np.testing.assert_allclose(row["action"][:2], [[0.0], [0.1]])


def test_read_zarr_reads_scalar_arrays(tmp_path: Path) -> None:
    path = tmp_path / "scalar.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset("version", data=np.asarray(3, dtype=np.int64), shape=())

    row = mdr.read_zarr(
        path,
        arrays={"version": "version"},
        file_path_column=None,
    ).take(1)[0]

    assert row["version"] == 3


def test_read_zarr_splits_arrays_by_row_ends(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)

    rows = mdr.read_zarr(
        path,
        arrays={
            "action": "data/action",
            "observation.state": "data/state",
            "frames": "data/rgb",
        },
        attrs={"task": "task"},
        row_ends="meta/episode_ends",
        file_path_column=None,
    ).take(2)

    assert [row["task"] for row in rows] == ["push tee", "push tee"]
    assert [len(row["action"]) for row in rows] == [2, 3]
    np.testing.assert_allclose(rows[0]["action"], [[0.0], [0.1]])
    np.testing.assert_allclose(rows[1]["action"], [[1.0], [1.1], [1.2]])
    np.testing.assert_array_equal(
        rows[0]["frames"],
        np.arange(2 * 4 * 4 * 3).reshape(2, 4, 4, 3),
    )


def test_read_zarr_plans_row_ends_with_num_shards(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)

    pipeline = mdr.read_zarr(
        path,
        arrays={"action": "data/action"},
        row_ends="meta/episode_ends",
        num_shards=2,
        file_path_column=None,
    )

    shards = pipeline.source.list_shards()
    ranges = [cast(RowRangeDescriptor, shard.descriptor) for shard in shards]
    assert [(item.start, item.end) for item in ranges] == [
        (0, 1),
        (1, 2),
    ]

    rows = [cast(Row, row) for row in pipeline.source.read_shard(shards[1])]
    assert len(rows) == 1
    assert rows[0]["index"] == 1
    np.testing.assert_allclose(rows[0]["action"], [[1.0], [1.1], [1.2]])


def test_read_zarr_allows_attrs_only_reads(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)

    row = mdr.read_zarr(
        path,
        arrays={},
        attrs={"task": "task"},
        file_path_column=None,
    ).take(1)[0]

    assert list(row) == ["task"]
    assert row["task"] == "push tee"


def test_read_zarr_rejects_duplicate_output_names(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)

    with pytest.raises(ValueError, match="duplicate output names"):
        mdr.read_zarr(
            path,
            arrays={"task": "data/action"},
            attrs={"task": "task"},
            file_path_column=None,
        )


def test_read_zarr_rejects_discovered_array_attr_collisions(tmp_path: Path) -> None:
    path = tmp_path / "collision.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset("task", data=np.asarray([1], dtype=np.int64))
    root.attrs["task"] = "push tee"

    pipeline = mdr.read_zarr(path, attrs={"task": "task"}, file_path_column=None)

    with pytest.raises(ValueError, match="duplicate output names"):
        pipeline.take(1)


def test_read_zarr_rejects_reserved_index_output_name(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)

    with pytest.raises(ValueError, match="reserved output names"):
        mdr.read_zarr(
            path,
            arrays={"index": "data/action"},
            row_ends="meta/episode_ends",
            file_path_column=None,
        )


def test_read_zarr_rejects_duplicate_metadata_column_names(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)

    with pytest.raises(ValueError, match="must be distinct"):
        mdr.read_zarr(
            path,
            row_ends="meta/episode_ends",
            file_path_column="metadata",
            index_column="metadata",
        )


def test_read_zarr_rejects_missing_selected_paths(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)

    with pytest.raises(KeyError, match="Zarr array not found"):
        mdr.read_zarr(
            path,
            arrays={"missing": "data/missing"},
            file_path_column=None,
        ).take(1)


def test_read_zarr_rejects_missing_selected_attrs(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)

    with pytest.raises(KeyError, match="Zarr attr not found"):
        mdr.read_zarr(
            path,
            arrays={},
            attrs={"missing_attr": "missing_attr"},
            file_path_column=None,
        ).take(1)


def test_read_zarr_split_leading_axis_emits_aligned_windows(tmp_path: Path) -> None:
    path = tmp_path / "windows.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset(
        "data/action",
        data=np.arange(5, dtype=np.float32).reshape(5, 1),
        chunks=(1, 1),
    )
    root.create_dataset(
        "data/rgb",
        data=np.arange(5 * 4 * 4 * 3, dtype=np.uint8).reshape(5, 4, 4, 3),
        chunks=(2, 4, 4, 3),
    )

    pipeline = mdr.read_zarr(
        path,
        arrays={"action": "data/action", "rgb": "data/rgb"},
        split_leading_axis=True,
        target_shard_bytes=96,
        file_path_column=None,
    )

    shards = pipeline.source.list_shards()
    ranges = [cast(RowRangeDescriptor, shard.descriptor) for shard in shards]
    assert [(item.start, item.end) for item in ranges] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]

    rows = pipeline.take(3)

    assert [row["index"] for row in rows] == [0, 1, 2]
    assert [len(row["action"]) for row in rows] == [2, 2, 1]
    np.testing.assert_allclose(rows[1]["action"], [[2.0], [3.0]])


def test_read_zarr_split_leading_axis_requires_aligned_lengths(tmp_path: Path) -> None:
    path = tmp_path / "misaligned.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset("data/action", data=np.zeros((5, 1), dtype=np.float32))
    root.create_dataset("data/state", data=np.zeros((4, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="same leading dimension"):
        mdr.read_zarr(
            path,
            arrays={"action": "data/action", "state": "data/state"},
            split_leading_axis=True,
            file_path_column=None,
        ).take(1)


def test_read_zarr_rejects_non_monotonic_row_ends(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)
    root = zarr.open_group(str(path), mode="a")
    root["meta/episode_ends"][:] = np.asarray([3, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="row_ends must be monotonic"):
        mdr.read_zarr(
            path,
            arrays={"action": "data/action"},
            row_ends="meta/episode_ends",
            file_path_column=None,
        ).take(1)


def test_read_zarr_rejects_out_of_range_row_ends(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)
    root = zarr.open_group(str(path), mode="a")
    root["meta/episode_ends"][:] = np.asarray([2, 6], dtype=np.int64)

    with pytest.raises(ValueError, match="row_ends exceed"):
        mdr.read_zarr(
            path,
            arrays={"action": "data/action"},
            row_ends="meta/episode_ends",
            file_path_column=None,
        ).take(2)


def test_read_zarr_rejects_short_row_ends(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)
    root = zarr.open_group(str(path), mode="a")
    root["meta/episode_ends"][:] = np.asarray([2, 4], dtype=np.int64)

    with pytest.raises(ValueError, match="end before leading dimension"):
        mdr.read_zarr(
            path,
            arrays={"action": "data/action"},
            row_ends="meta/episode_ends",
            file_path_column=None,
        ).take(2)


def test_zarr_to_robot_rows_and_lerobot_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    lerobot_out = tmp_path / "lerobot"
    _write_policy_zarr(path)

    (
        mdr.read_zarr(
            path,
            arrays={
                "action": "data/action",
                "observation.state": "data/state",
                "frames": "data/rgb",
            },
            attrs={"dataset_id": "dataset_id", "task": "task"},
            row_ends="meta/episode_ends",
            file_path_column=None,
        )
        .to_robot_rows(
            episode_id_key="index",
            task_key="task",
            action_key="action",
            state_key="observation.state",
            video_keys={"observation.images.front": "frames"},
            fps=10,
            robot_type="pusht",
        )
        .write_lerobot(str(lerobot_out), max_video_prepare_in_flight=1)
        .launch_local(
            name="zarr-to-lerobot", num_workers=1, rundir=str(tmp_path / "run1")
        )
    )

    episodes = [
        cast(RoboticsRow, row)
        for row in mdr.read_lerobot(str(lerobot_out)).materialize()
    ]
    episodes.sort(key=lambda episode: int(episode.episode_id))
    assert [episode.num_frames for episode in episodes] == [2, 3]
    assert [episode.task for episode in episodes] == ["push tee", "push tee"]
