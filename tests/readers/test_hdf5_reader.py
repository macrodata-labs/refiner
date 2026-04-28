from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pytest

from refiner.pipeline import read_hdf5
from refiner.pipeline.data import datatype
from refiner.pipeline.sources.readers.hdf5 import Hdf5Reader


def _write_demo_file(path: Path, *, task: bytes | str = "stack blocks") -> None:
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        demo_0 = data.create_group("demo_0")
        demo_0.attrs["task"] = task
        demo_0.create_dataset("actions", data=np.array([[1, 2], [3, 4]]))
        obs = demo_0.create_group("obs")
        obs.create_dataset("rgb", data=np.ones((2, 4, 4, 3), dtype=np.uint8))

        demo_1 = data.create_group("demo_1")
        demo_1.attrs["task"] = "place cup"
        demo_1.create_dataset("actions", data=np.array([[5, 6]]))
        demo_1.create_group("obs").create_dataset(
            "rgb", data=np.zeros((1, 4, 4, 3), dtype=np.uint8)
        )


def test_hdf5_reader_reads_one_row_per_matching_group() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo.hdf5"
        _write_demo_file(path, task=b"stack blocks")

        rows = read_hdf5(
            str(path),
            groups="/data/demo_*",
            datasets={"actions": "actions", "frames": "obs/rgb"},
            attrs={"task": "task"},
        ).take(10)

        assert [row["hdf5_group"] for row in rows] == ["/data/demo_0", "/data/demo_1"]
        assert rows[0]["file_path"] == str(path)
        assert rows[0]["task"] == "stack blocks"
        np.testing.assert_array_equal(rows[0]["actions"], np.array([[1, 2], [3, 4]]))
        assert rows[0]["frames"].shape == (2, 4, 4, 3)


def test_hdf5_reader_reads_root_group_by_default() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "root.h5"
        with h5py.File(path, "w") as f:
            f.attrs["task"] = "root task"
            f.create_dataset("actions", data=np.array([1, 2, 3]))

        row = read_hdf5(
            str(path),
            datasets={"actions": "actions"},
            attrs={"task": "task"},
            group_path_column=None,
            file_path_column=None,
        ).take(1)[0]

        assert row["task"] == "root task"
        np.testing.assert_array_equal(row["actions"], np.array([1, 2, 3]))
        assert set(row) == {"actions", "task"}


def test_hdf5_reader_plans_without_opening_hdf5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryDirectory() as tmp:
        first = Path(tmp) / "first.h5"
        second = Path(tmp) / "second.h5"
        _write_demo_file(first)
        _write_demo_file(second)

        def fail_open(*args, **kwargs):
            raise AssertionError("list_shards should not open HDF5 files")

        monkeypatch.setattr(h5py, "File", fail_open)

        shards = read_hdf5(
            [str(first), str(second)], target_shard_bytes=1
        ).source.list_shards()

        assert len(shards) == 2


def test_hdf5_reader_missing_skip_drops_group() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo.h5"
        _write_demo_file(path)

        rows = read_hdf5(
            str(path),
            groups="/data/demo_*",
            datasets={"actions": "actions", "missing": "does_not_exist"},
            missing="skip",
        ).take(10)

        assert rows == []


def test_hdf5_reader_schema_contains_only_dtype_overrides() -> None:
    reader = Hdf5Reader(
        "missing.h5",
        datasets={"frames": "frames"},
        dtypes={"frames": datatype.video_file()},
    )

    assert reader.schema is not None
    assert reader.schema.names == ["frames"]
