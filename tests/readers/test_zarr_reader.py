from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import zarr

import refiner as mdr
from refiner.robotics.row import RoboticsRow


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
            episode_id_key="dataset_id",
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
    assert [episode.num_frames for episode in episodes] == [2, 3]
    assert [episode.task for episode in episodes] == ["push tee", "push tee"]
