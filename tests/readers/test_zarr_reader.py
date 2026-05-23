from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Literal, cast

import numpy as np
import pytest
import zarr

import refiner as mdr
from refiner.robotics.row import RoboticsRow
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.shard import RowRangeDescriptor


def _open_test_zarr(path: Path, *, mode: Literal["r", "r+", "a", "w", "w-"]):
    kwargs: dict[str, Any] = {"mode": mode, "zarr_format": 2}
    try:
        return zarr.open_group(str(path), **kwargs)
    except TypeError:
        return zarr.open_group(str(path), mode=mode)


def _create_array(root, name: str, data, **kwargs):
    if hasattr(root, "create_array"):
        kwargs.pop("shape", None)
        return root.create_array(name, data=data, **kwargs)
    return root.create_dataset(name, data=data, **kwargs)


def _write_policy_zarr(path: Path) -> None:
    root = _open_test_zarr(path, mode="w")
    _create_array(
        root,
        "data/action",
        data=np.asarray([[0.0], [0.1], [1.0], [1.1], [1.2]], dtype=np.float32),
    )
    _create_array(
        root,
        "data/state",
        data=np.asarray([[10.0], [10.1], [20.0], [20.1], [20.2]], dtype=np.float32),
    )
    _create_array(
        root,
        "data/rgb",
        data=np.arange(5 * 4 * 4 * 3, dtype=np.uint8).reshape(5, 4, 4, 3),
    )
    _create_array(root, "meta/episode_ends", data=np.asarray([2, 5], dtype=np.int64))
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
    root = _open_test_zarr(path, mode="w")
    _create_array(root, "version", data=np.asarray(3, dtype=np.int64), shape=())

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


def test_read_zarr_reads_zip_store(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)
    zip_path = Path(shutil.make_archive(str(path), "zip", root_dir=path))

    row = mdr.read_zarr(
        str(zip_path),
        arrays={"action": "data/action", "frames": "data/rgb"},
        row_ends="meta/episode_ends",
    ).take(1)[0]

    assert row["file_path"] == str(zip_path)
    assert row["action"].shape == (2, 1)
    assert row["frames"].shape == (2, 4, 4, 3)


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
    root = _open_test_zarr(path, mode="w")
    _create_array(root, "task", data=np.asarray([1], dtype=np.int64))
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


def test_read_zarr_split_leading_axis_emits_one_row_per_index(tmp_path: Path) -> None:
    path = tmp_path / "leading_axis.zarr"
    root = _open_test_zarr(path, mode="w")
    _create_array(
        root,
        "data/action",
        data=np.arange(5, dtype=np.float32).reshape(5, 1),
        chunks=(1, 1),
    )
    _create_array(
        root,
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
    assert [row["action"].shape for row in rows] == [(1, 1), (1, 1), (1, 1)]
    assert [row["rgb"].shape for row in rows] == [(1, 4, 4, 3)] * 3
    np.testing.assert_allclose(rows[1]["action"], [[1.0]])


def test_read_zarr_split_leading_axis_uses_row_size(tmp_path: Path) -> None:
    path = tmp_path / "leading_axis_rows.zarr"
    root = _open_test_zarr(path, mode="w")
    _create_array(
        root,
        "data/action",
        data=np.arange(6, dtype=np.float32).reshape(6, 1),
        chunks=(2, 1),
    )

    rows = mdr.read_zarr(
        path,
        arrays={"action": "data/action"},
        split_leading_axis=True,
        leading_axis_row_size=2,
        file_path_column=None,
    ).take(3)

    assert [row["index"] for row in rows] == [0, 1, 2]
    assert [row["action"].shape for row in rows] == [(2, 1), (2, 1), (2, 1)]
    np.testing.assert_allclose(rows[1]["action"], [[2.0], [3.0]])


def test_read_zarr_split_leading_axis_uses_row_batch_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "leading_axis_batched.zarr"
    root = _open_test_zarr(path, mode="w")
    _create_array(
        root,
        "data/action",
        data=np.arange(6, dtype=np.float32).reshape(6, 1),
        chunks=(1, 1),
    )

    pipeline = mdr.read_zarr(
        path,
        arrays={"action": "data/action"},
        split_leading_axis=True,
        row_batch_size=2,
        file_path_column=None,
    )
    source = cast(Any, pipeline.source)
    shards = source.list_shards()
    calls: list[tuple[int | None, int | None]] = []
    read_arrays = source._read_arrays

    def record_read_arrays(arrays, *, start=None, end=None):
        calls.append((start, end))
        return read_arrays(arrays, start=start, end=end)

    monkeypatch.setattr(source, "_read_arrays", record_read_arrays)
    rows = list(source.read_shard(shards[0]))

    assert [row["index"] for row in rows] == list(range(6))
    np.testing.assert_allclose(rows[5]["action"], [[5.0]])
    assert calls == [(0, 2), (2, 4), (4, 6)]


def test_read_zarr_split_leading_axis_uses_dominant_array_chunks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image_dominant_chunks.zarr"
    root = _open_test_zarr(path, mode="w")
    _create_array(
        root,
        "data/action",
        data=np.arange(10, dtype=np.float32).reshape(10, 1),
        chunks=(10, 1),
    )
    _create_array(
        root,
        "data/rgb",
        data=np.zeros((10, 4, 4, 3), dtype=np.uint8),
        chunks=(2, 4, 4, 3),
    )

    pipeline = mdr.read_zarr(
        path,
        arrays={"action": "data/action", "rgb": "data/rgb"},
        split_leading_axis=True,
        target_shard_bytes=160,
        file_path_column=None,
    )

    shards = pipeline.source.list_shards()
    ranges = [cast(RowRangeDescriptor, shard.descriptor) for shard in shards]
    assert [(item.start, item.end) for item in ranges] == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
        (8, 10),
    ]


def test_read_zarr_split_leading_axis_requires_aligned_lengths(tmp_path: Path) -> None:
    path = tmp_path / "misaligned.zarr"
    root = _open_test_zarr(path, mode="w")
    _create_array(root, "data/action", data=np.zeros((5, 1), dtype=np.float32))
    _create_array(root, "data/state", data=np.zeros((4, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="same leading dimension"):
        mdr.read_zarr(
            path,
            arrays={"action": "data/action", "state": "data/state"},
            split_leading_axis=True,
            file_path_column=None,
        ).take(1)


def test_read_zarr_split_leading_axis_requires_full_rows(tmp_path: Path) -> None:
    path = tmp_path / "partial-leading-axis-row.zarr"
    root = _open_test_zarr(path, mode="w")
    _create_array(root, "data/action", data=np.zeros((5, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="divisible by row size"):
        mdr.read_zarr(
            path,
            arrays={"action": "data/action"},
            split_leading_axis=True,
            leading_axis_row_size=2,
            file_path_column=None,
        ).take(1)


def test_read_zarr_leading_axis_row_size_requires_split_mode(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)

    with pytest.raises(ValueError, match="requires split_leading_axis"):
        mdr.read_zarr(
            path,
            arrays={"action": "data/action"},
            leading_axis_row_size=2,
        )


def test_read_zarr_rejects_invalid_row_batch_size(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)

    with pytest.raises(ValueError, match="row_batch_size"):
        mdr.read_zarr(
            path,
            arrays={"action": "data/action"},
            split_leading_axis=True,
            row_batch_size=0,
        )


def test_read_zarr_rejects_non_monotonic_row_ends(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)
    root = _open_test_zarr(path, mode="a")
    root["meta/episode_ends"][:] = np.asarray([3, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="row_ends must be monotonic"):
        mdr.read_zarr(
            path,
            arrays={"action": "data/action"},
            row_ends="meta/episode_ends",
            file_path_column=None,
        ).take(1)


def test_read_zarr_rejects_non_integer_row_ends(tmp_path: Path) -> None:
    path = tmp_path / "float-row-ends.zarr"
    root = _open_test_zarr(path, mode="w")
    _create_array(root, "data/action", data=np.zeros((5, 1), dtype=np.float32))
    _create_array(root, "meta/episode_ends", data=np.asarray([2.5, 5.0]))

    with pytest.raises(ValueError, match="integer offsets"):
        mdr.read_zarr(
            path,
            arrays={"action": "data/action"},
            row_ends="meta/episode_ends",
            file_path_column=None,
        ).take(1)


def test_read_zarr_rejects_out_of_range_row_ends(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)
    root = _open_test_zarr(path, mode="a")
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
    root = _open_test_zarr(path, mode="a")
    root["meta/episode_ends"][:] = np.asarray([2, 4], dtype=np.int64)

    with pytest.raises(ValueError, match="end before leading dimension"):
        mdr.read_zarr(
            path,
            arrays={"action": "data/action"},
            row_ends="meta/episode_ends",
            file_path_column=None,
        ).take(2)


def test_read_zarr_rejects_scalar_arrays_in_row_ends_mode(tmp_path: Path) -> None:
    path = tmp_path / "scalar-row-ends.zarr"
    root = _open_test_zarr(path, mode="w")
    _create_array(root, "version", data=np.asarray(3, dtype=np.int64), shape=())
    _create_array(root, "meta/episode_ends", data=np.asarray([1], dtype=np.int64))

    with pytest.raises(ValueError, match="must have a leading dimension"):
        mdr.read_zarr(
            path,
            arrays={"version": "version"},
            row_ends="meta/episode_ends",
            file_path_column=None,
        ).take(1)


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
