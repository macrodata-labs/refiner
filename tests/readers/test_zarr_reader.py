from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Literal, cast

from fsspec.implementations.memory import MemoryFileSystem
import numpy as np
import pytest
import zarr

import refiner as mdr
from refiner.io.datafolder import DataFolder
from refiner.io import DataFile
from refiner.robotics.row import RoboticsRow
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.shard import RowRangeDescriptor
from refiner.pipeline.sinks.reducer.zarr import ZarrReducerSink
from refiner.pipeline.sinks.zarr import ZarrSink
from refiner.worker.context import set_active_run_context, worker_token_for
from refiner.worker.lifecycle import FinalizedShardWorker, RuntimeLifecycle


class _FinalizedWorkersRuntime:
    def __init__(self, rows: list[FinalizedShardWorker]) -> None:
        self._rows = rows

    def finalized_workers(
        self, *, stage_index: int | None = None
    ) -> list[FinalizedShardWorker]:
        assert stage_index == 0
        return self._rows


class _EmptyVideoSource:
    def clipped(self, **_kwargs):
        return self

    async def iter_frames(self):
        if False:
            yield None

    async def iter_numpy_frames(self):
        if False:
            yield None

    async def iter_frame_windows(self, **_kwargs):
        if False:
            yield None

    async def write_to(self, writer, **_kwargs):
        raise NotImplementedError


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


def _write_part_zarr(path: Path, arrays: dict[str, np.ndarray]) -> None:
    root = _open_test_zarr(path, mode="w")
    for name, data in arrays.items():
        _create_array(root, name, data=data)


def _write_video(path: Path, *, num_frames: int = 3, fps: int = 5) -> None:
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 4
        stream.height = 4
        stream.pix_fmt = "yuv420p"

        for value in range(num_frames):
            frame = av.VideoFrame.from_ndarray(
                np.full((4, 4, 3), value, dtype=np.uint8),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)


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


def test_read_zarr_reads_multiple_inputs(tmp_path: Path) -> None:
    first = tmp_path / "first.zarr"
    second = tmp_path / "second.zarr"
    first_root = _open_test_zarr(first, mode="w")
    second_root = _open_test_zarr(second, mode="w")
    _create_array(
        first_root,
        "data/action",
        data=np.asarray([[1.0]], dtype=np.float32),
    )
    _create_array(
        second_root,
        "data/action",
        data=np.asarray([[2.0]], dtype=np.float32),
    )

    rows = mdr.read_zarr(
        [first, second],
        arrays={"action": "data/action"},
        file_path_column=None,
    ).take(2)

    assert [row["action"].item() for row in rows] == [1.0, 2.0]

    shards = mdr.read_zarr(
        [first, second],
        arrays={"action": "data/action"},
        file_path_column=None,
    ).source.list_shards()
    assert [(shard.start_key, shard.end_key) for shard in shards] == [
        ("0", "0"),
        ("1", "1"),
    ]


def test_read_zarr_reads_folder_glob(tmp_path: Path) -> None:
    first = tmp_path / "first.zarr"
    second = tmp_path / "second.zarr"
    first_root = _open_test_zarr(first, mode="w")
    second_root = _open_test_zarr(second, mode="w")
    _create_array(first_root, "data/action", data=np.asarray([[1.0]], dtype=np.float32))
    _create_array(
        second_root,
        "data/action",
        data=np.asarray([[2.0]], dtype=np.float32),
    )

    rows = mdr.read_zarr(
        str(tmp_path / "*.zarr"),
        arrays={"action": "data/action"},
        file_path_column=None,
    ).take(2)

    assert [row["action"].item() for row in rows] == [1.0, 2.0]


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


def test_read_zarr_reads_zip_datafolder(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)
    zip_path = Path(shutil.make_archive(str(path), "zip", root_dir=path))

    row = mdr.read_zarr(
        DataFolder(str(zip_path)),
        arrays={"action": "data/action"},
        row_ends="meta/episode_ends",
    ).take(1)[0]

    assert row["file_path"] == str(zip_path)
    assert row["action"].shape == (2, 1)


def test_read_zarr_reads_remote_store(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)

    fs = MemoryFileSystem()
    remote_root = "/policy.zarr"
    for source_path in path.rglob("*"):
        if source_path.is_file():
            relative_path = source_path.relative_to(path)
            remote_path = f"{remote_root}/{relative_path}"
            fs.makedirs(str(Path(remote_path).parent), exist_ok=True)
            with source_path.open("rb") as src, fs.open(remote_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

    row = mdr.read_zarr(
        (remote_root, fs),
        arrays={"action": "data/action"},
        row_ends="meta/episode_ends",
    ).take(1)[0]

    assert row["file_path"] == "memory:///policy.zarr"
    assert row["action"].shape == (2, 1)


def test_read_zarr_reads_remote_zip_without_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)
    zip_path = Path(shutil.make_archive(str(path), "zip", root_dir=path))

    fs = MemoryFileSystem()
    remote_path = "/policy.zarr.zip"
    with zip_path.open("rb") as src, fs.open(remote_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    open_calls: list[dict[str, Any]] = []
    opened_files: list[int] = []
    closed_files: list[int] = []
    original_open = fs.open

    def record_open(path, mode="rb", **kwargs):
        if path == remote_path and mode == "rb":
            open_calls.append(kwargs)
            file = original_open(path, mode=mode, **kwargs)
            opened_files.append(id(file))
            original_close = file.close

            def record_close():
                closed_files.append(id(file))
                original_close()

            file.close = record_close
            return file
        return original_open(path, mode=mode, **kwargs)

    monkeypatch.setattr(fs, "open", record_open)

    row = mdr.read_zarr(
        (remote_path, fs),
        arrays={"action": "data/action"},
        row_ends="meta/episode_ends",
    ).take(1)[0]

    assert row["action"].shape == (2, 1)
    assert open_calls
    assert open_calls[0]["cache_type"] == "none"
    assert set(closed_files) == set(opened_files)


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


def test_read_zarr_rejects_selecting_row_ends_as_output_array(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    _write_policy_zarr(path)

    with pytest.raises(ValueError, match="cannot also be row_ends"):
        mdr.read_zarr(
            path,
            arrays={"episode_ends": "meta/episode_ends"},
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


def test_read_zarr_rejects_empty_row_ends_for_nonempty_arrays(tmp_path: Path) -> None:
    path = tmp_path / "empty-row-ends.zarr"
    root = _open_test_zarr(path, mode="w")
    _create_array(root, "data/action", data=np.zeros((2, 1), dtype=np.float32))
    _create_array(root, "meta/episode_ends", data=np.asarray([], dtype=np.int64))

    with pytest.raises(ValueError, match="end before leading dimension"):
        mdr.read_zarr(
            path,
            arrays={"action": "data/action"},
            row_ends="meta/episode_ends",
            file_path_column=None,
        ).take(1)


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


def test_write_zarr_roundtrips_lerobot_rows(tmp_path: Path) -> None:
    path = tmp_path / "policy.zarr"
    lerobot_out = tmp_path / "lerobot"
    zarr_out = tmp_path / "roundtrip.zarr"
    _write_policy_zarr(path)

    (
        mdr.read_zarr(
            path,
            arrays={
                "action": "data/action",
                "observation.state": "data/state",
                "frames": "data/rgb",
            },
            attrs={"task": "task"},
            row_ends="meta/episode_ends",
            file_path_column=None,
        )
        .to_robot_rows(
            episode_id_key="row_index",
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

    (
        mdr.read_lerobot(str(lerobot_out))
        .write_zarr(
            str(zarr_out),
            arrays={
                "data/action": "action",
                "data/state": "observation.state",
            },
            reduce_to_single_store=False,
        )
        .launch_local(
            name="lerobot-to-zarr", num_workers=1, rundir=str(tmp_path / "run2")
        )
    )

    rows = [
        mdr.read_zarr(
            zarr_store,
            arrays={
                "action": "data/action",
                "state": "data/state",
                "episode_ends": "meta/episode_ends",
            },
            file_path_column=None,
        ).take(1)[0]
        for zarr_store in sorted(zarr_out.glob("*.zarr"))
    ]

    episode_ends = np.concatenate([row["episode_ends"] for row in rows]).tolist()
    assert episode_ends[-1] == 5
    assert sorted(np.diff([0, *episode_ends]).tolist()) == [2, 3]
    action = np.concatenate([row["action"] for row in rows])
    state = np.concatenate([row["state"] for row in rows])
    assert action.shape == (5, 1)
    np.testing.assert_allclose(
        np.sort(state.reshape(-1)),
        np.asarray([10.0, 10.1, 20.0, 20.1, 20.2]),
    )


def test_write_zarr_can_reduce_to_single_store(tmp_path: Path) -> None:
    zarr_out = tmp_path / "single.zarr"

    (
        mdr.from_items(
            [
                {"action": [[0.0], [0.1]], "state": [[1.0], [1.1]], "task": "push"},
                {"action": [[0.2]], "state": [[1.2]], "task": "push"},
            ],
            items_per_shard=1,
        )
        .write_zarr(
            str(zarr_out),
            arrays={
                "data/action": "action",
                "data/state": "state",
            },
            attrs={"task": "task"},
            reduce_to_single_store=True,
        )
        .launch_local(
            name="zarr-single-store", num_workers=1, rundir=str(tmp_path / "run")
        )
    )

    row = mdr.read_zarr(
        zarr_out,
        arrays={
            "action": "data/action",
            "state": "data/state",
            "episode_ends": "meta/episode_ends",
        },
        attrs={"task": "task"},
        file_path_column=None,
    ).take(1)[0]

    np.testing.assert_allclose(row["action"], [[0.0], [0.1], [0.2]])
    np.testing.assert_allclose(row["state"], [[1.0], [1.1], [1.2]])
    assert row["episode_ends"].tolist() == [2, 3]
    assert row["task"] == "push"
    assert not (zarr_out / "_parts").exists()


def test_write_zarr_single_store_preserves_empty_payload_arrays(tmp_path: Path) -> None:
    zarr_out = tmp_path / "single-empty-payload.zarr"

    (
        mdr.from_items([{"action": np.empty((0, 2), dtype=np.float32)}])
        .write_zarr(
            str(zarr_out),
            arrays={"data/action": "action"},
            reduce_to_single_store=True,
        )
        .launch_local(
            name="zarr-single-empty-payload",
            num_workers=1,
            rundir=str(tmp_path / "run-empty-payload"),
        )
    )

    root = _open_test_zarr(zarr_out, mode="r")
    assert "data/action" in root
    assert root["data/action"].shape == (0, 2)
    assert root["meta/episode_ends"][:].tolist() == [0]


def test_write_zarr_rejects_store_template_without_worker_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="store_template requires fields"):
        ZarrSink(str(tmp_path / "template.zarr"), store_template="{shard_id}.zarr")


def test_write_zarr_rejects_unsupported_store_template_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="store_template only supports"):
        ZarrSink(
            str(tmp_path / "extra-template.zarr"),
            store_template="{shard_id}__w{worker_id}__{part}.zarr",
        )
    with pytest.raises(ValueError, match="store_template only supports"):
        ZarrSink(
            str(tmp_path / "format-template.zarr"),
            store_template="{shard_id:>12}__w{worker_id}.zarr",
        )


def test_write_zarr_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not contain"):
        ZarrSink(
            str(tmp_path / "escape.zarr"),
            store_template="../escape/{shard_id}__w{worker_id}.zarr",
        )
    with pytest.raises(ValueError, match="must be relative"):
        ZarrSink(
            str(tmp_path / "absolute.zarr"),
            store_template="/tmp/{shard_id}__w{worker_id}.zarr",
        )
    with pytest.raises(ValueError, match="must not contain"):
        ZarrSink(
            str(tmp_path / "array-escape.zarr"),
            arrays={"../action": "action"},
        )
    with pytest.raises(ValueError, match="must not be empty"):
        ZarrSink(
            str(tmp_path / "empty-array-path.zarr"),
            arrays={"": "action"},
        )
    with pytest.raises(ValueError, match="must not be empty"):
        ZarrSink(
            str(tmp_path / "empty-episode-ends.zarr"),
            arrays={"data/action": "action"},
            episode_ends_path="",
        )


def test_write_zarr_rejects_rendered_path_traversal(tmp_path: Path) -> None:
    with set_active_run_context(
        job_id="local",
        stage_index=0,
        worker_id="worker-a",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime([])),
    ):
        with pytest.raises(ValueError, match="must not contain"):
            ZarrSink(
                str(tmp_path / "rendered-escape.zarr"),
                arrays={"data/action": "action"},
            ).write_block([DictRow({"action": [[1.0]]}, shard_id="../escape")])


def test_write_zarr_rejects_reserved_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reserved root"):
        ZarrSink(
            str(tmp_path / "reserved-array.zarr"),
            arrays={"_parts/action": "action"},
        )
    with pytest.raises(ValueError, match="reserved root"):
        ZarrSink(
            str(tmp_path / "reserved-episode-ends.zarr"),
            arrays={"data/action": "action"},
            episode_ends_path="_parts/episode_ends",
        )
    with pytest.raises(ValueError, match="reserved root"):
        ZarrSink(
            str(tmp_path / "reserved-template.zarr"),
            store_template="_refiner/{shard_id}__w{worker_id}.zarr",
        )


def test_write_zarr_rejects_empty_array_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="arrays must not be empty"):
        ZarrSink(str(tmp_path / "empty-arrays.zarr"), arrays={})


def test_write_zarr_rejects_empty_default_robotics_arrays(tmp_path: Path) -> None:
    rows = list(
        mdr.from_items([{"episode_id": "episode-1"}]).to_robot_rows(
            episode_id_key="episode_id",
            action_key=None,
            state_key=None,
            timestamp_key=None,
        )
    )

    with pytest.raises(ValueError, match="inferred no default robotics arrays"):
        ZarrSink(str(tmp_path / "empty-defaults.zarr")).write_block(rows)


def test_write_zarr_single_store_replace_ignores_stale_parts(tmp_path: Path) -> None:
    zarr_out = tmp_path / "single-replace.zarr"

    (
        mdr.from_items([{"action": [[0.0]]}], items_per_shard=1)
        .write_zarr(
            str(zarr_out),
            arrays={"data/action": "action"},
            reduce_to_single_store=True,
        )
        .launch_local(
            name="zarr-single-replace-first",
            num_workers=1,
            rundir=str(tmp_path / "run-first"),
        )
    )

    stale_part = zarr_out / "_parts" / "old__wold.zarr"
    stale_part.mkdir(parents=True)
    (stale_part / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")

    (
        mdr.from_items([{"action": [[1.0]]}], items_per_shard=1)
        .write_zarr(
            str(zarr_out),
            arrays={"data/action": "action"},
            reduce_to_single_store=True,
        )
        .launch_local(
            name="zarr-single-replace-second",
            num_workers=1,
            rundir=str(tmp_path / "run-second"),
        )
    )

    row = mdr.read_zarr(
        zarr_out,
        arrays={"action": "data/action"},
        file_path_column=None,
    ).take(1)[0]
    np.testing.assert_allclose(row["action"], [[1.0]])
    assert not stale_part.exists()


def test_write_zarr_sharded_replace_removes_single_store_payload_and_parts(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "sharded-replaces-single-store.zarr"

    (
        mdr.from_items([{"action": [[0.0]]}], items_per_shard=1)
        .write_zarr(
            str(zarr_out),
            arrays={"data/action": "action"},
            reduce_to_single_store=True,
        )
        .launch_local(
            name="zarr-sharded-replace-first",
            num_workers=1,
            rundir=str(tmp_path / "run-sharded-replace-first"),
        )
    )
    stale_part = zarr_out / "_parts" / "old__wold.zarr"
    stale_part.mkdir(parents=True)
    (stale_part / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")

    (
        mdr.from_items([{"action": [[1.0], [2.0]]}], items_per_shard=1)
        .write_zarr(
            str(zarr_out),
            arrays={"data/action": "action"},
            reduce_to_single_store=False,
        )
        .launch_local(
            name="zarr-sharded-replace-second",
            num_workers=1,
            rundir=str(tmp_path / "run-sharded-replace-second"),
        )
    )

    assert not (zarr_out / "data").exists()
    assert not (zarr_out / "meta").exists()
    assert not (zarr_out / "_parts").exists()
    stores = sorted(zarr_out.glob("*.zarr"))
    assert len(stores) == 1
    row = mdr.read_zarr(
        stores[0],
        arrays={
            "action": "data/action",
            "episode_ends": "meta/episode_ends",
        },
        file_path_column=None,
    ).take(1)[0]
    np.testing.assert_allclose(row["action"], [[1.0], [2.0]])
    assert row["episode_ends"].tolist() == [2]


def test_write_zarr_sharded_replace_clears_payload_under_store_prefix(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "sharded-replaces-nested-single-store.zarr"

    (
        mdr.from_items([{"action": [[0.0]]}], items_per_shard=1)
        .write_zarr(
            str(zarr_out),
            arrays={"split/action": "action"},
            reduce_to_single_store=True,
        )
        .launch_local(
            name="zarr-sharded-nested-replace-first",
            num_workers=1,
            rundir=str(tmp_path / "run-sharded-nested-replace-first"),
        )
    )

    (
        mdr.from_items([{"action": [[1.0], [2.0]]}], items_per_shard=1)
        .write_zarr(
            str(zarr_out),
            arrays={"data/action": "action"},
            store_template="split/{shard_id}__w{worker_id}.zarr",
            reduce_to_single_store=False,
        )
        .launch_local(
            name="zarr-sharded-nested-replace-second",
            num_workers=1,
            rundir=str(tmp_path / "run-sharded-nested-replace-second"),
        )
    )

    assert not (zarr_out / "split" / "action").exists()
    assert not (zarr_out / "meta").exists()
    stores = sorted((zarr_out / "split").glob("*.zarr"))
    assert len(stores) == 1
    row = mdr.read_zarr(
        stores[0],
        arrays={"action": "data/action"},
        file_path_column=None,
    ).take(1)[0]
    np.testing.assert_allclose(row["action"], [[1.0], [2.0]])


def test_write_zarr_non_reduced_cleanup_rejects_missing_finalized_store(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "sharded-missing-finalized-store.zarr"
    runtime = _FinalizedWorkersRuntime(
        [FinalizedShardWorker(shard_id="shard-a", worker_id="worker-a")]
    )
    with set_active_run_context(
        job_id="local",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, runtime),
    ):
        with pytest.raises(ValueError, match="Zarr store is missing"):
            ZarrReducerSink(
                str(zarr_out),
                store_template="{shard_id}__w{worker_id}.zarr",
                reduce_to_single_store=False,
            ).write_block([DictRow({}, shard_id="reduce")])


def test_write_zarr_empty_shard_completion_replaces_stale_store(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "empty-shard-replaces-stale-store.zarr"
    worker_id = "worker-a"
    stale = zarr_out / f"shard-a__w{worker_token_for(worker_id)}.zarr"
    _write_part_zarr(stale, {"data/action": np.asarray([[9.0]], dtype=np.float32)})

    with set_active_run_context(
        job_id="local",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime([])),
    ):
        ZarrSink(
            str(zarr_out),
            arrays={"data/action": "action"},
            reduce_to_single_store=False,
        ).on_shard_complete("shard-a")

    root = _open_test_zarr(stale, mode="r")
    assert not list(root.array_keys())
    assert not list(root.group_keys())


def test_write_zarr_non_reduced_cleanup_keeps_empty_stores_retryable(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "sharded-empty-cleanup-retry.zarr"
    worker_id = "worker-a"
    empty_store = zarr_out / f"shard-a__w{worker_token_for(worker_id)}.zarr"
    _open_test_zarr(empty_store, mode="w")

    runtime = _FinalizedWorkersRuntime(
        [FinalizedShardWorker(shard_id="shard-a", worker_id=worker_id)]
    )
    for _ in range(2):
        with set_active_run_context(
            job_id="local",
            stage_index=1,
            worker_id="reducer",
            worker_name=None,
            runtime_lifecycle=cast(RuntimeLifecycle, runtime),
        ):
            ZarrReducerSink(
                str(zarr_out),
                store_template="{shard_id}__w{worker_id}.zarr",
                reduce_to_single_store=False,
            ).write_block([DictRow({}, shard_id="reduce")])

    assert empty_store.exists()


def test_write_zarr_rejects_sharded_schema_drift_after_cleanup(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "sharded-schema-drift.zarr"
    first_worker = "worker-a"
    second_worker = "worker-b"
    first = zarr_out / f"shard-a__w{worker_token_for(first_worker)}.zarr"
    second = zarr_out / f"shard-b__w{worker_token_for(second_worker)}.zarr"
    _write_part_zarr(
        first,
        {
            "data/action": np.asarray([[0.0]], dtype=np.float32),
            "meta/episode_ends": np.asarray([1], dtype=np.int64),
        },
    )
    _write_part_zarr(
        second,
        {
            "data/action": np.asarray([[1.0]], dtype=np.float32),
            "data/state": np.asarray([[2.0]], dtype=np.float32),
            "meta/episode_ends": np.asarray([1], dtype=np.int64),
        },
    )
    reducer = ZarrSink(
        str(zarr_out),
        arrays={"data/action": "action"},
        reduce_to_single_store=False,
    ).build_reducer()
    assert reducer is not None
    runtime = _FinalizedWorkersRuntime(
        [
            FinalizedShardWorker(
                shard_id="shard-a",
                worker_id=first_worker,
                global_ordinal=0,
            ),
            FinalizedShardWorker(
                shard_id="shard-b",
                worker_id=second_worker,
                global_ordinal=1,
            ),
        ]
    )

    with set_active_run_context(
        job_id="local",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, runtime),
    ):
        with pytest.raises(ValueError, match="same arrays"):
            reducer.write_block([DictRow({}, shard_id="reduce")])


def test_write_zarr_single_store_skips_empty_shards(tmp_path: Path) -> None:
    zarr_out = tmp_path / "single-empty-shards.zarr"

    (
        mdr.from_items(
            [{"action": [[0.0]]}, {"action": [[0.1]]}],
            items_per_shard=1,
        )
        .filter(lambda row: False)
        .write_zarr(
            str(zarr_out),
            arrays={"data/action": "action"},
            reduce_to_single_store=True,
        )
        .launch_local(
            name="zarr-single-empty-shards",
            num_workers=2,
            rundir=str(tmp_path / "run-empty"),
        )
    )

    assert not (zarr_out / "_parts").exists()


def test_write_zarr_single_store_empty_replace_ignores_stale_done_marker(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "single-empty-replace-stale-done.zarr"

    (
        mdr.from_items([{"action": [[1.0]]}], items_per_shard=1)
        .write_zarr(
            str(zarr_out),
            arrays={"data/action": "action"},
            reduce_to_single_store=True,
        )
        .launch_local(
            name="zarr-single-stale-done-first",
            num_workers=1,
            rundir=str(tmp_path / "run-stale-done-first"),
        )
    )

    (
        mdr.from_items([{"action": [[2.0]]}], items_per_shard=1)
        .filter(lambda row: False)
        .write_zarr(
            str(zarr_out),
            arrays={"data/action": "action"},
            reduce_to_single_store=True,
        )
        .launch_local(
            name="zarr-single-stale-done-second",
            num_workers=1,
            rundir=str(tmp_path / "run-stale-done-second"),
        )
    )

    root = _open_test_zarr(zarr_out, mode="r")
    assert "data/action" not in root
    assert not (zarr_out / "_parts").exists()


def test_write_zarr_single_store_skips_mixed_empty_shards(tmp_path: Path) -> None:
    zarr_out = tmp_path / "single-mixed-empty-shards.zarr"

    (
        mdr.from_items(
            [{"action": [[0.0]]}, {"action": [[1.0]]}],
            items_per_shard=1,
        )
        .filter(lambda row: float(row["action"][0][0]) > 0.0)
        .write_zarr(
            str(zarr_out),
            arrays={"data/action": "action"},
            reduce_to_single_store=True,
        )
        .launch_local(
            name="zarr-single-mixed-empty-shards",
            num_workers=2,
            rundir=str(tmp_path / "run-mixed-empty"),
        )
    )

    row = mdr.read_zarr(
        zarr_out,
        arrays={
            "action": "data/action",
            "episode_ends": "meta/episode_ends",
        },
        file_path_column=None,
    ).take(1)[0]
    np.testing.assert_allclose(row["action"], [[1.0]])
    assert row["episode_ends"].tolist() == [1]
    assert not (zarr_out / "_parts").exists()


def test_write_zarr_single_store_offsets_batched_episode_ends(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "single-batched-episode-ends.zarr"
    workers = ["worker-a", "worker-b"]
    for shard_id, worker in enumerate(workers):
        part = (
            zarr_out / "_parts" / f"shard-{shard_id}__w{worker_token_for(worker)}.zarr"
        )
        _write_part_zarr(
            part,
            {
                "data/action": np.arange(2, dtype=np.float32).reshape(2, 1),
                "meta/episode_ends": np.asarray([1, 2], dtype=np.int64),
            },
        )
    runtime = _FinalizedWorkersRuntime(
        [
            FinalizedShardWorker(
                shard_id=f"shard-{shard_id}",
                worker_id=worker,
                global_ordinal=shard_id,
            )
            for shard_id, worker in enumerate(workers)
        ]
    )
    with set_active_run_context(
        job_id="local",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, runtime),
    ):
        ZarrReducerSink(
            str(zarr_out),
            store_template="{shard_id}__w{worker_id}.zarr",
            episode_ends_path="meta/episode_ends",
            array_chunk_bytes=8,
            reduce_to_single_store=True,
        ).write_block([DictRow({}, shard_id="reduce")])

    root = _open_test_zarr(zarr_out, mode="r")
    assert root["meta/episode_ends"][:].tolist() == [1, 2, 3, 4]


def test_write_zarr_single_store_rejects_inconsistent_part_payloads(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "single-inconsistent-parts.zarr"
    first_worker = "worker-a"
    second_worker = "worker-b"
    first_part = (
        zarr_out / "_parts" / f"shard-a__w{worker_token_for(first_worker)}.zarr"
    )
    second_part = (
        zarr_out / "_parts" / f"shard-b__w{worker_token_for(second_worker)}.zarr"
    )
    _write_part_zarr(
        first_part,
        {
            "data/action": np.asarray([[0.0]], dtype=np.float32),
            "meta/episode_ends": np.asarray([1], dtype=np.int64),
        },
    )
    _write_part_zarr(
        second_part,
        {
            "data/action": np.asarray([[1.0]], dtype=np.float32),
            "data/state": np.asarray([[2.0]], dtype=np.float32),
            "meta/episode_ends": np.asarray([1], dtype=np.int64),
        },
    )

    runtime = _FinalizedWorkersRuntime(
        [
            FinalizedShardWorker(
                shard_id="shard-a",
                worker_id=first_worker,
                global_ordinal=0,
            ),
            FinalizedShardWorker(
                shard_id="shard-b",
                worker_id=second_worker,
                global_ordinal=1,
            ),
        ]
    )
    with set_active_run_context(
        job_id="local",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, runtime),
    ):
        with pytest.raises(ValueError, match="same payload arrays"):
            ZarrReducerSink(
                str(zarr_out),
                store_template="{shard_id}__w{worker_id}.zarr",
                episode_ends_path="meta/episode_ends",
                array_chunk_bytes=1024,
                reduce_to_single_store=True,
            ).write_block([DictRow({}, shard_id="reduce")])
    assert first_part.exists()
    assert second_part.exists()


def test_write_zarr_single_store_rejects_part_missing_episode_ends(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "single-missing-episode-ends.zarr"
    worker = "worker-a"
    part = zarr_out / "_parts" / f"shard-a__w{worker_token_for(worker)}.zarr"
    _write_part_zarr(
        part,
        {"data/action": np.asarray([[0.0]], dtype=np.float32)},
    )
    runtime = _FinalizedWorkersRuntime(
        [
            FinalizedShardWorker(
                shard_id="shard-a",
                worker_id=worker,
                global_ordinal=0,
            )
        ]
    )
    with set_active_run_context(
        job_id="local",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, runtime),
    ):
        with pytest.raises(ValueError, match="meta/episode_ends"):
            ZarrReducerSink(
                str(zarr_out),
                store_template="{shard_id}__w{worker_id}.zarr",
                episode_ends_path="meta/episode_ends",
                array_chunk_bytes=1024,
                reduce_to_single_store=True,
            ).write_block([DictRow({}, shard_id="reduce")])


def test_write_zarr_single_store_rejects_part_row_end_mismatch(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "single-row-end-mismatch.zarr"
    worker = "worker-a"
    part = zarr_out / "_parts" / f"shard-a__w{worker_token_for(worker)}.zarr"
    _write_part_zarr(
        part,
        {
            "data/action": np.asarray([[0.0], [1.0]], dtype=np.float32),
            "meta/episode_ends": np.asarray([1], dtype=np.int64),
        },
    )
    runtime = _FinalizedWorkersRuntime(
        [
            FinalizedShardWorker(
                shard_id="shard-a",
                worker_id=worker,
                global_ordinal=0,
            )
        ]
    )
    with set_active_run_context(
        job_id="local",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, runtime),
    ):
        with pytest.raises(ValueError, match="episode_ends final value"):
            ZarrReducerSink(
                str(zarr_out),
                store_template="{shard_id}__w{worker_id}.zarr",
                episode_ends_path="meta/episode_ends",
                array_chunk_bytes=1024,
                reduce_to_single_store=True,
            ).write_block([DictRow({}, shard_id="reduce")])


def test_write_zarr_single_store_rejects_missing_finalized_part(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "single-missing-part.zarr"
    _write_part_zarr(
        zarr_out,
        {
            "data/action": np.asarray([[9.0]], dtype=np.float32),
            "meta/episode_ends": np.asarray([1], dtype=np.int64),
        },
    )
    runtime = _FinalizedWorkersRuntime(
        [
            FinalizedShardWorker(
                shard_id="shard-a",
                worker_id="worker-a",
                global_ordinal=0,
            )
        ]
    )

    with set_active_run_context(
        job_id="local",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, runtime),
    ):
        with pytest.raises(ValueError, match="part store is missing"):
            ZarrReducerSink(
                str(zarr_out),
                store_template="{shard_id}__w{worker_id}.zarr",
                episode_ends_path="meta/episode_ends",
                array_chunk_bytes=1024,
                reduce_to_single_store=True,
            ).write_block([DictRow({}, shard_id="reduce")])

    row = mdr.read_zarr(
        zarr_out,
        arrays={"action": "data/action", "episode_ends": "meta/episode_ends"},
        file_path_column=None,
    ).take(1)[0]
    np.testing.assert_allclose(row["action"], [[9.0]])
    assert row["episode_ends"].tolist() == [1]


def test_write_zarr_single_store_removes_parts_only_on_completion(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "single-complete-cleans-parts.zarr"
    worker_id = "worker-a"
    part = zarr_out / "_parts" / f"shard-a__w{worker_token_for(worker_id)}.zarr"
    _write_part_zarr(
        part,
        {
            "data/action": np.asarray([[9.0]], dtype=np.float32),
            "meta/episode_ends": np.asarray([1], dtype=np.int64),
        },
    )
    runtime = _FinalizedWorkersRuntime(
        [
            FinalizedShardWorker(
                shard_id="shard-a",
                worker_id=worker_id,
                global_ordinal=0,
            )
        ]
    )

    with set_active_run_context(
        job_id="local",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, runtime),
    ):
        reducer = ZarrReducerSink(
            str(zarr_out),
            store_template="{shard_id}__w{worker_id}.zarr",
            episode_ends_path="meta/episode_ends",
            array_chunk_bytes=1024,
            reduce_to_single_store=True,
        )
        reducer.write_block([DictRow({}, shard_id="reduce")])
        assert part.exists()
        reducer.on_shard_finalized("reduce")

    row = mdr.read_zarr(
        zarr_out,
        arrays={"action": "data/action", "episode_ends": "meta/episode_ends"},
        file_path_column=None,
    ).take(1)[0]
    np.testing.assert_allclose(row["action"], [[9.0]])
    assert row["episode_ends"].tolist() == [1]
    assert not (zarr_out / "_parts").exists()


def test_write_zarr_single_store_zero_shard_replace_clears_existing_output(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "single-zero-shard-replace.zarr"
    _write_part_zarr(
        zarr_out,
        {
            "data/action": np.asarray([[9.0]], dtype=np.float32),
            "meta/episode_ends": np.asarray([1], dtype=np.int64),
        },
    )
    runtime = _FinalizedWorkersRuntime([])

    with set_active_run_context(
        job_id="local",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, runtime),
    ):
        ZarrReducerSink(
            str(zarr_out),
            store_template="{shard_id}__w{worker_id}.zarr",
            episode_ends_path="meta/episode_ends",
            array_chunk_bytes=1024,
            reduce_to_single_store=True,
        ).write_block([DictRow({}, shard_id="reduce")])

    root = _open_test_zarr(zarr_out, mode="r")
    assert "data/action" not in root


def test_write_zarr_single_store_parts_are_resume_stable(tmp_path: Path) -> None:
    zarr_out = tmp_path / "single-resume-stable.zarr"
    worker_id = "original-worker"
    part = zarr_out / "_parts" / f"shard-a__w{worker_token_for(worker_id)}.zarr"
    _write_part_zarr(
        part,
        {
            "data/action": np.asarray([[4.0]], dtype=np.float32),
            "meta/episode_ends": np.asarray([1], dtype=np.int64),
        },
    )
    runtime = _FinalizedWorkersRuntime(
        [
            FinalizedShardWorker(
                shard_id="shard-a",
                worker_id=worker_id,
                global_ordinal=0,
            )
        ]
    )

    with set_active_run_context(
        job_id="resumed-job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, runtime),
    ):
        ZarrReducerSink(
            str(zarr_out),
            store_template="{shard_id}__w{worker_id}.zarr",
            episode_ends_path="meta/episode_ends",
            array_chunk_bytes=1024,
            reduce_to_single_store=True,
        ).write_block([DictRow({}, shard_id="reduce")])

    row = mdr.read_zarr(
        zarr_out,
        arrays={"action": "data/action", "episode_ends": "meta/episode_ends"},
        file_path_column=None,
    ).take(1)[0]
    np.testing.assert_allclose(row["action"], [[4.0]])
    assert row["episode_ends"].tolist() == [1]


def test_write_zarr_single_store_replace_clears_root_attrs(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "single-replace-attrs.zarr"
    root = _open_test_zarr(zarr_out, mode="w")
    root.attrs["task"] = "old"

    (
        mdr.from_items([{"action": [[1.0]]}], items_per_shard=1)
        .write_zarr(
            str(zarr_out),
            arrays={"data/action": "action"},
            reduce_to_single_store=True,
        )
        .launch_local(
            name="zarr-single-replace-attrs",
            num_workers=1,
            rundir=str(tmp_path / "run-replace-attrs"),
        )
    )

    root = _open_test_zarr(zarr_out, mode="r")
    assert dict(root.attrs) == {}


def test_write_zarr_single_store_rejects_part_attr_drift_before_clearing(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "single-attr-drift.zarr"
    _write_part_zarr(
        zarr_out,
        {"data/action": np.asarray([[99.0]], dtype=np.float32)},
    )
    root = _open_test_zarr(zarr_out, mode="r+")
    root.attrs["task"] = "old"

    first_worker = "worker-a"
    second_worker = "worker-b"
    first_part = (
        zarr_out / "_parts" / f"shard-a__w{worker_token_for(first_worker)}.zarr"
    )
    second_part = (
        zarr_out / "_parts" / f"shard-b__w{worker_token_for(second_worker)}.zarr"
    )
    _write_part_zarr(
        first_part,
        {
            "data/action": np.asarray([[0.0]], dtype=np.float32),
            "meta/episode_ends": np.asarray([1], dtype=np.int64),
        },
    )
    _write_part_zarr(
        second_part,
        {
            "data/action": np.asarray([[1.0]], dtype=np.float32),
            "meta/episode_ends": np.asarray([1], dtype=np.int64),
        },
    )
    _open_test_zarr(first_part, mode="r+").attrs["task"] = "first"
    _open_test_zarr(second_part, mode="r+").attrs["task"] = "second"

    runtime = _FinalizedWorkersRuntime(
        [
            FinalizedShardWorker(
                shard_id="shard-a",
                worker_id=first_worker,
                global_ordinal=0,
            ),
            FinalizedShardWorker(
                shard_id="shard-b",
                worker_id=second_worker,
                global_ordinal=1,
            ),
        ]
    )
    with set_active_run_context(
        job_id="local",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, runtime),
    ):
        with pytest.raises(ValueError, match="attrs differ"):
            ZarrReducerSink(
                str(zarr_out),
                store_template="{shard_id}__w{worker_id}.zarr",
                episode_ends_path="meta/episode_ends",
                array_chunk_bytes=1024,
                reduce_to_single_store=True,
            ).write_block([DictRow({}, shard_id="reduce")])

    root = _open_test_zarr(zarr_out, mode="r")
    np.testing.assert_allclose(root["data/action"][:], [[99.0]])
    assert dict(root.attrs) == {"task": "old"}


def test_write_zarr_single_store_rejects_part_dtype_drift(
    tmp_path: Path,
) -> None:
    zarr_out = tmp_path / "single-dtype-drift.zarr"
    first_worker = "worker-a"
    second_worker = "worker-b"
    first_part = (
        zarr_out / "_parts" / f"shard-a__w{worker_token_for(first_worker)}.zarr"
    )
    second_part = (
        zarr_out / "_parts" / f"shard-b__w{worker_token_for(second_worker)}.zarr"
    )
    _write_part_zarr(
        first_part,
        {
            "data/action": np.asarray([[0.0]], dtype=np.float32),
            "meta/episode_ends": np.asarray([1], dtype=np.int64),
        },
    )
    _write_part_zarr(
        second_part,
        {
            "data/action": np.asarray([[1.0]], dtype=np.float64),
            "meta/episode_ends": np.asarray([1], dtype=np.int64),
        },
    )

    runtime = _FinalizedWorkersRuntime(
        [
            FinalizedShardWorker(
                shard_id="shard-a",
                worker_id=first_worker,
                global_ordinal=0,
            ),
            FinalizedShardWorker(
                shard_id="shard-b",
                worker_id=second_worker,
                global_ordinal=1,
            ),
        ]
    )
    with set_active_run_context(
        job_id="local",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, runtime),
    ):
        with pytest.raises(ValueError, match="matching dtypes"):
            ZarrReducerSink(
                str(zarr_out),
                store_template="{shard_id}__w{worker_id}.zarr",
                episode_ends_path="meta/episode_ends",
                array_chunk_bytes=1024,
                reduce_to_single_store=True,
            ).write_block([DictRow({}, shard_id="reduce")])
    assert first_part.exists()
    assert second_part.exists()


def test_write_zarr_rejects_rows_missing_inferred_default_arrays(
    tmp_path: Path,
) -> None:
    output = tmp_path / "missing-default-array.zarr"
    rows = list(
        mdr.from_items(
            [
                {"action": [[0.0]], "observation.state": [[1.0]]},
                {"action": [[0.1]]},
            ]
        ).to_robot_rows(
            action_key="action",
            state_key="observation.state",
            timestamp_key=None,
        )
    )

    with pytest.raises(ValueError, match="default arrays changed"):
        ZarrSink(str(output)).write_block(rows)


def test_write_zarr_rejects_late_default_arrays(tmp_path: Path) -> None:
    output = tmp_path / "late-default-array.zarr"
    rows = list(
        mdr.from_items(
            [
                {"action": [[0.0]]},
                {"action": [[0.1]], "observation.state": [[1.1]]},
            ]
        ).to_robot_rows(
            action_key="action",
            state_key="observation.state",
            timestamp_key=None,
        )
    )

    with pytest.raises(ValueError, match="default arrays changed"):
        ZarrSink(str(output)).write_block(rows)


def test_write_zarr_rejects_episode_ends_path_collision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="collides with episode_ends_path"):
        ZarrSink(
            str(tmp_path / "collision.zarr"),
            arrays={"meta/episode_ends": "action"},
        )
    with pytest.raises(ValueError, match="collides with episode_ends_path"):
        ZarrSink(
            str(tmp_path / "normalized-collision.zarr"),
            arrays={"meta//episode_ends": "action"},
        )


def test_write_zarr_rejects_shape_drift_before_appending_bad_row(
    tmp_path: Path,
) -> None:
    output = tmp_path / "shape-mismatch.zarr"
    rows: list[Row] = [
        DictRow({"action": [[0.0]], "state": [[1.0, 2.0]]}, shard_id="shard"),
        DictRow({"action": [[0.1]], "state": [[1.1]]}, shard_id="shard"),
    ]

    with pytest.raises(ValueError, match="matching trailing shapes"):
        ZarrSink(
            str(output),
            arrays={
                "data/action": "action",
                "data/state": "state",
            },
            reduce_to_single_store=False,
        ).write_block(rows)

    zarr_store = next(output.glob("*.zarr"))
    row = mdr.read_zarr(
        zarr_store,
        arrays={"action": "data/action", "state": "data/state"},
        file_path_column=None,
    ).take(1)[0]
    assert row["action"].shape == (1, 1)
    assert row["state"].shape == (1, 2)


def test_write_zarr_materializes_frame_array_videos(tmp_path: Path) -> None:
    output = tmp_path / "video.zarr"
    frames = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    rows = list(
        mdr.from_items(
            [{"episode_id": "episode-1", "frames": frames, "action": [[0.0], [0.1]]}]
        ).to_robot_rows(
            episode_id_key="episode_id",
            action_key="action",
            state_key=None,
            timestamp_key=None,
            video_keys={"observation.images.front": "frames"},
            fps=10,
        )
    )

    ZarrSink(
        str(output),
        arrays={
            "data/action": "action",
            "data/rgb": "observation.images.front",
        },
        reduce_to_single_store=False,
    ).write_block(rows)

    zarr_store = next(output.glob("*.zarr"))
    row = mdr.read_zarr(
        zarr_store,
        arrays={"action": "data/action", "rgb": "data/rgb"},
        file_path_column=None,
    ).take(1)[0]
    np.testing.assert_array_equal(row["rgb"], frames)
    np.testing.assert_allclose(row["action"], [[0.0], [0.1]])


def test_write_zarr_defaults_include_robotics_videos(tmp_path: Path) -> None:
    output = tmp_path / "default-video.zarr"
    frames = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    rows = list(
        mdr.from_items(
            [{"episode_id": "episode-1", "frames": frames, "action": [[0.0], [0.1]]}]
        ).to_robot_rows(
            episode_id_key="episode_id",
            action_key="action",
            state_key=None,
            timestamp_key=None,
            video_keys={"observation.images.front": "frames"},
            fps=10,
        )
    )

    ZarrSink(str(output), reduce_to_single_store=False).write_block(rows)

    zarr_store = next(output.glob("*.zarr"))
    row = mdr.read_zarr(
        zarr_store,
        arrays={
            "action": "data/action",
            "front": "data/observation.images.front",
        },
        file_path_column=None,
    ).take(1)[0]
    np.testing.assert_array_equal(row["front"], frames)
    np.testing.assert_allclose(row["action"], [[0.0], [0.1]])


def test_write_zarr_rejects_empty_frame_array_videos(tmp_path: Path) -> None:
    output = tmp_path / "empty-video.zarr"
    frames = np.empty((0, 4, 5, 3), dtype=np.uint8)
    rows = list(
        mdr.from_items([{"episode_id": "episode-1", "frames": frames}]).to_robot_rows(
            episode_id_key="episode_id",
            action_key=None,
            state_key=None,
            timestamp_key=None,
            video_keys={"observation.images.front": "frames"},
            fps=10,
        )
    )

    with pytest.raises(ValueError, match="produced no frames"):
        ZarrSink(
            str(output),
            arrays={"data/rgb": "observation.images.front"},
            reduce_to_single_store=False,
        ).write_block(rows)


def test_write_zarr_uses_byte_budgeted_chunks_for_large_rows(tmp_path: Path) -> None:
    output = tmp_path / "video-chunks.zarr"
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    rows = list(
        mdr.from_items(
            [{"episode_id": "episode-1", "frames": frames, "action": [[0.0], [0.1]]}]
        ).to_robot_rows(
            episode_id_key="episode_id",
            action_key="action",
            state_key=None,
            timestamp_key=None,
            video_keys={"observation.images.front": "frames"},
            fps=10,
        )
    )

    ZarrSink(
        str(output),
        arrays={
            "data/action": "action",
            "data/rgb": "observation.images.front",
        },
        array_chunk_bytes=50,
        reduce_to_single_store=False,
    ).write_block(rows)

    root = _open_test_zarr(next(output.glob("*.zarr")), mode="r")
    assert root["data/rgb"].chunks == (1, 4, 4, 3)


def test_write_zarr_caps_low_dimensional_initial_chunks(tmp_path: Path) -> None:
    output = tmp_path / "small-array-chunks.zarr"
    ZarrSink(
        str(output),
        arrays={"data/action": "action"},
        reduce_to_single_store=False,
    ).write_block(
        [
            DictRow(
                {"action": np.asarray([[1.0]], dtype=np.float32)},
                shard_id="shard",
            )
        ]
    )

    root = _open_test_zarr(next(output.glob("*.zarr")), mode="r")
    assert root["data/action"].chunks == (1024, 1)


def test_write_zarr_streams_encoded_videos(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "encoded-video.zarr"
    _write_video(source, num_frames=3, fps=5)

    rows = list(
        mdr.from_items(
            [
                {
                    "episode_id": "episode-1",
                    "clip": mdr.video.VideoFile(DataFile.resolve(source)),
                    "action": [[0.0], [0.1], [0.2]],
                }
            ]
        ).to_robot_rows(
            episode_id_key="episode_id",
            action_key="action",
            state_key=None,
            timestamp_key=None,
            video_keys={"observation.images.front": "clip"},
        )
    )

    ZarrSink(
        str(output),
        arrays={
            "data/action": "action",
            "data/rgb": "observation.images.front",
        },
        video_frame_batch_size=2,
        reduce_to_single_store=False,
    ).write_block(rows)

    zarr_store = next(output.glob("*.zarr"))
    row = mdr.read_zarr(
        zarr_store,
        arrays={"action": "data/action", "rgb": "data/rgb"},
        file_path_column=None,
    ).take(1)[0]
    assert row["rgb"].shape == (3, 4, 4, 3)
    assert row["rgb"].dtype == np.uint8
    np.testing.assert_allclose(row["action"], [[0.0], [0.1], [0.2]])


def test_write_zarr_rejects_empty_encoded_video_source(tmp_path: Path) -> None:
    output = tmp_path / "empty-encoded-video.zarr"

    rows = list(
        mdr.from_items(
            [{"episode_id": "episode-1", "clip": _EmptyVideoSource()}]
        ).to_robot_rows(
            episode_id_key="episode_id",
            action_key=None,
            state_key=None,
            timestamp_key=None,
            video_keys={"observation.images.front": "clip"},
        )
    )

    with pytest.raises(ValueError, match="produced no frames"):
        ZarrSink(
            str(output),
            arrays={"data/rgb": "observation.images.front"},
        ).write_block(rows)


def test_write_zarr_rejects_video_length_mismatch_before_final_append(
    tmp_path: Path,
) -> None:
    output = tmp_path / "video-length-mismatch.zarr"
    frames = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    rows = list(
        mdr.from_items(
            [{"episode_id": "episode-1", "frames": frames, "action": [[0.0], [0.1]]}]
        ).to_robot_rows(
            episode_id_key="episode_id",
            action_key="action",
            state_key=None,
            timestamp_key=None,
            video_keys={"observation.images.front": "frames"},
            fps=10,
        )
    )

    with pytest.raises(ValueError, match="matching lengths"):
        ZarrSink(
            str(output),
            arrays={
                "data/action": "action",
                "data/rgb": "observation.images.front",
            },
            video_frame_batch_size=2,
            reduce_to_single_store=False,
        ).write_block(rows)

    zarr_store = next(output.glob("*.zarr"))
    root = _open_test_zarr(zarr_store, mode="r")
    assert "data/action" not in root
    assert "data/rgb" not in root
    assert "__tmp" not in root
