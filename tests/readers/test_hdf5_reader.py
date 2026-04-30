from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import pyarrow as pa
import pytest

from refiner.pipeline import read_hdf5
from refiner.pipeline.data import datatype
from refiner.pipeline.sources.readers.hdf5 import Hdf5Reader


def _rows_to_table(block, schema: pa.Schema | None = None) -> pa.Table:
    assert isinstance(block, list)
    assert block
    return block[0].tabular_type.from_rows(block, schema=schema).table


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


def test_hdf5_reader_reads_one_row_per_matching_group(tmp_path: Path) -> None:
    path = tmp_path / "demo.hdf5"
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
    assert isinstance(rows[0]["actions"], np.ndarray)
    np.testing.assert_array_equal(rows[0]["actions"], np.array([[1, 2], [3, 4]]))
    assert np.asarray(rows[0]["frames"]).shape == (2, 4, 4, 3)


def test_hdf5_reader_reads_root_group_by_default(tmp_path: Path) -> None:
    path = tmp_path / "root.h5"
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _write_demo_file(first)
    _write_demo_file(second)

    def fail_open(*args, **kwargs):
        raise AssertionError("list_shards should not open HDF5 files")

    monkeypatch.setattr(h5py, "File", fail_open)

    shards = read_hdf5(
        [str(first), str(second)], target_shard_bytes=1
    ).source.list_shards()

    assert len(shards) == 2


def test_hdf5_reader_keeps_files_atomic_when_more_shards_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "demo.h5"
    _write_demo_file(path)
    warnings: list[str] = []

    monkeypatch.setattr(
        "refiner.pipeline.sources.readers.base.logger.warning",
        lambda message, *args: warnings.append(message.format(*args)),
    )
    shards = read_hdf5(str(path), num_shards=2).source.list_shards()

    assert len(shards) == 1
    assert warnings == [
        "read_hdf5 requested 2 shards, but this reader keeps files atomic and "
        "only found 1 input files; planned 1 shards."
    ]


def test_hdf5_reader_explicit_num_shards_is_exact_for_files(tmp_path: Path) -> None:
    paths = [tmp_path / f"demo-{i}.h5" for i in range(3)]
    for path in paths:
        _write_demo_file(path)

    shards = read_hdf5(
        [str(path) for path in paths],
        num_shards=3,
    ).source.list_shards()

    assert len(shards) == 3


def test_hdf5_reader_accepts_single_dataset_path(tmp_path: Path) -> None:
    path = tmp_path / "demo.h5"
    _write_demo_file(path)

    row = read_hdf5(
        str(path),
        groups="/data/demo_0",
        datasets="actions",
        file_path_column=None,
    ).take(1)[0]

    assert set(row) == {"hdf5_group", "actions"}
    np.testing.assert_array_equal(row["actions"], np.array([[1, 2], [3, 4]]))


def test_hdf5_reader_allows_different_dataset_shapes(tmp_path: Path) -> None:
    path = tmp_path / "different-shapes.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("data").create_group("demo_0")
        group.create_dataset("actions", data=np.ones((2, 7)))
        group.create_dataset("frames", data=np.ones((3, 4, 4, 3), dtype=np.uint8))

    row = read_hdf5(
        str(path),
        groups="/data/demo_0",
        datasets={"actions": "actions", "frames": "frames"},
    ).take(1)[0]

    assert row["actions"].shape == (2, 7)
    assert row["frames"].shape == (3, 4, 4, 3)


def test_hdf5_reader_reads_scalar_datasets(tmp_path: Path) -> None:
    path = tmp_path / "scalar.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("data").create_group("demo_0")
        group.create_dataset("actions", data=np.ones((2, 7)))
        group.create_dataset("task_id", data=3)

    row = read_hdf5(
        str(path),
        groups="/data/demo_0",
        datasets={"actions": "actions", "task_id": "task_id"},
        file_path_column=None,
    ).take(1)[0]

    assert row["task_id"] == 3


def test_hdf5_reader_rejects_absolute_dataset_paths() -> None:
    with pytest.raises(ValueError, match="must be relative"):
        Hdf5Reader("missing.h5", datasets={"actions": "/data/actions"})


def test_hdf5_reader_rejects_duplicate_derived_selection_names() -> None:
    with pytest.raises(ValueError, match="duplicate name 'rgb'"):
        Hdf5Reader(
            "missing.h5",
            datasets=["left/rgb", "right/rgb"],
        )


def test_hdf5_reader_rejects_output_name_collisions() -> None:
    with pytest.raises(ValueError, match="duplicate name 'x'"):
        Hdf5Reader("missing.h5", datasets={"x": "data"}, attrs={"x": "attr"})

    with pytest.raises(ValueError, match="duplicate name 'hdf5_group'"):
        Hdf5Reader("missing.h5", datasets={"hdf5_group": "data"})

    with pytest.raises(ValueError, match="must be distinct"):
        Hdf5Reader("missing.h5", file_path_column="path", group_path_column="path")


def test_hdf5_reader_preserves_binary_byte_scalars(tmp_path: Path) -> None:
    path = tmp_path / "binary.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("data").create_group("demo_0")
        group.attrs["payload"] = b"\xff\xfe"

    row = read_hdf5(
        str(path),
        groups="/data/demo_0",
        attrs={"payload": "payload"},
        file_path_column=None,
    ).take(1)[0]

    assert row["payload"] == b"\xff\xfe"


def test_hdf5_reader_preserves_fixed_byte_datasets(tmp_path: Path) -> None:
    path = tmp_path / "binary-dataset.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("data").create_group("demo_0")
        group.create_dataset("payload", data=np.array([b"\xff\xfe"], dtype="S2"))

    pipeline = read_hdf5(
        str(path),
        groups="/data/demo_0",
        datasets={"payload": "payload"},
        file_path_column=None,
    ).write_parquet(tmp_path / "out")

    blocks = list(pipeline.execute(pipeline.source.read()))
    table = _rows_to_table(blocks[0], schema=pipeline.source.schema)
    assert table.schema.field("payload").type == pa.list_(pa.binary())
    assert table.column("payload")[0].as_py() == [b"\xff\xfe"]


def test_hdf5_reader_decodes_utf8_byte_array_datasets(tmp_path: Path) -> None:
    path = tmp_path / "strings.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("data").create_group("demo_0")
        group.create_dataset(
            "labels",
            data=np.array(["pick", "place"], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )

    row = read_hdf5(
        str(path),
        groups="/data/demo_0",
        datasets={"labels": "labels"},
        file_path_column=None,
    ).take(1)[0]

    assert row["labels"] == ["pick", "place"]


def test_hdf5_reader_arrays_survive_tabular_steps(tmp_path: Path) -> None:
    path = tmp_path / "demo.h5"
    _write_demo_file(path)

    row = (
        read_hdf5(
            str(path),
            groups="/data/demo_0",
            datasets={"actions": "actions", "frames": "obs/rgb"},
            file_path_column=None,
        )
        .select("actions")
        .take(1)[0]
    )

    assert row["actions"] == [[1, 2], [3, 4]]


def test_hdf5_reader_missing_drop_row_drops_group(tmp_path: Path) -> None:
    path = tmp_path / "demo.h5"
    _write_demo_file(path)

    rows = read_hdf5(
        str(path),
        groups="/data/demo_*",
        datasets={"actions": "actions", "missing": "does_not_exist"},
        missing_policy="drop_row",
    ).take(10)

    assert rows == []


def test_hdf5_reader_missing_set_null_uses_explicit_dtypes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "demo.h5"
    _write_demo_file(path)
    with h5py.File(path, "a") as f:
        f["data"]["demo_1"].attrs["maybe_task"] = "place cup"

    pipeline = read_hdf5(
        str(path),
        groups="/data/demo_*",
        datasets={"actions": "actions"},
        attrs={"maybe_task": "maybe_task"},
        missing_policy="set_null",
        file_path_column=None,
        dtypes={"maybe_task": datatype.string()},
    ).write_parquet(tmp_path / "out")

    blocks = list(pipeline.execute(pipeline.source.read()))

    tables = [_rows_to_table(block, schema=pipeline.source.schema) for block in blocks]
    assert [table.schema.field("maybe_task").type for table in tables] == [
        pa.string(),
    ]
    assert [table.schema.field("actions").type for table in tables] == [
        pa.list_(pa.list_(pa.int64())),
    ]


def test_hdf5_reader_missing_set_null_uses_explicit_dtypes_across_files(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _write_demo_file(first)
    _write_demo_file(second)
    with h5py.File(second, "a") as f:
        f["data"]["demo_0"].attrs["maybe_task"] = "stack blocks"

    pipeline = read_hdf5(
        [str(first), str(second)],
        groups="/data/demo_0",
        attrs={"maybe_task": "maybe_task"},
        missing_policy="set_null",
        num_shards=1,
        file_path_column=None,
        dtypes={"maybe_task": datatype.string()},
    ).write_parquet(tmp_path / "out")

    blocks = list(pipeline.execute(pipeline.source.read()))

    tables = [_rows_to_table(block, schema=pipeline.source.schema) for block in blocks]
    assert [table.schema.field("maybe_task").type for table in tables] == [
        pa.string(),
    ]


def test_hdf5_reader_missing_set_null_all_null_column(tmp_path: Path) -> None:
    path = tmp_path / "demo.h5"
    _write_demo_file(path)

    row = read_hdf5(
        str(path),
        groups="/data/demo_0",
        attrs={"maybe_task": "maybe_task"},
        missing_policy="set_null",
        file_path_column=None,
    ).take(1)[0]

    assert row["maybe_task"] is None


def test_hdf5_reader_group_miss_emits_no_rows(tmp_path: Path) -> None:
    path = tmp_path / "demo.h5"
    _write_demo_file(path)

    rows = read_hdf5(str(path), groups="/missing/demo_*").take(1)
    exact_rows = read_hdf5(str(path), groups=["/missing", "/also_missing"]).take(1)

    assert rows == []
    assert exact_rows == []


def test_hdf5_reader_rejects_globs_in_group_lists() -> None:
    with pytest.raises(ValueError, match="single glob string"):
        Hdf5Reader("missing.h5", groups=["/data/demo_*", "/other"])


def test_hdf5_reader_rejects_multiple_recursive_group_globs() -> None:
    with pytest.raises(ValueError, match="at most one"):
        Hdf5Reader("missing.h5", groups="/data/**/**")


def test_hdf5_reader_normalizes_relative_glob_groups(tmp_path: Path) -> None:
    path = tmp_path / "demo.h5"
    _write_demo_file(path)
    with h5py.File(path, "a") as f:
        nested = f.create_group("nested").create_group("data").create_group("demo_2")
        nested.create_dataset("actions", data=np.array([[7, 8]]))

    rows = read_hdf5(
        str(path),
        groups="data/demo_*",
        datasets={"actions": "actions"},
        file_path_column=None,
    ).take(10)

    assert [row["hdf5_group"] for row in rows] == ["/data/demo_0", "/data/demo_1"]


def test_hdf5_reader_recursive_group_glob_matches_zero_or_more_segments(
    tmp_path: Path,
) -> None:
    path = tmp_path / "demo.h5"
    _write_demo_file(path)
    with h5py.File(path, "a") as f:
        nested = f["data"].create_group("nested").create_group("deeper")
        nested_demo = nested.create_group("demo_2")
        nested_demo.create_dataset("actions", data=np.array([[7, 8]]))

    rows = read_hdf5(
        str(path),
        groups="/data/**/demo_*",
        datasets={"actions": "actions"},
        file_path_column=None,
    ).take(10)

    assert [row["hdf5_group"] for row in rows] == [
        "/data/demo_0",
        "/data/demo_1",
        "/data/nested/deeper/demo_2",
    ]


def test_hdf5_reader_recursive_group_glob_skips_link_cycles(tmp_path: Path) -> None:
    path = tmp_path / "cycle.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("data")
        group.create_dataset("actions", data=np.array([1]))
        group["self"] = group

    rows = read_hdf5(
        str(path),
        groups="/data/**",
        datasets={"actions": "actions"},
        file_path_column=None,
    ).take(10)

    assert [row["hdf5_group"] for row in rows] == ["/data"]


def test_hdf5_reader_recursive_group_glob_keeps_hard_link_aliases(
    tmp_path: Path,
) -> None:
    path = tmp_path / "aliases.h5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        episode = data.create_group("left").create_group("episode")
        episode.create_dataset("actions", data=np.array([1]))
        data["right"] = data["left"]

    rows = read_hdf5(
        str(path),
        groups="/data/**/episode",
        datasets={"actions": "actions"},
        file_path_column=None,
    ).take(10)

    assert [row["hdf5_group"] for row in rows] == [
        "/data/left/episode",
        "/data/right/episode",
    ]


def test_hdf5_reader_schema_contains_only_dtype_overrides() -> None:
    reader = Hdf5Reader(
        "missing.h5",
        datasets={"frames": "frames"},
        dtypes={"frames": datatype.video_path()},
    )

    assert reader.schema is not None
    assert reader.schema.names == ["frames"]


def test_hdf5_reader_describe_includes_dtype_values() -> None:
    description = Hdf5Reader(
        "missing.h5",
        datasets={"frames": "frames"},
        dtypes={"frames": datatype.video_path()},
    ).describe()

    assert description["dtypes"] == {
        "frames": {"type": "string", "metadata": {"asset_type": "video"}}
    }
