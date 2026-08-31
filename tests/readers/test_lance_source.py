from __future__ import annotations

import pyarrow as pa
import pytest
from fsspec.implementations.memory import MemoryFileSystem

from refiner import AddColumns, load_lance, read_blob
from refiner.pipeline.data import datatype
from refiner.pipeline.data.shard import SOURCE_ROW_ID_COLUMN, RowRangeDescriptor
from refiner.pipeline.sources.lance import LanceSource
from refiner.pipeline.data.tabular import Tabular


def test_load_lance_rejects_explicit_shard_count_above_limit() -> None:
    with pytest.raises(ValueError, match=r"num_shards must be <= 10,000"):
        LanceSource("unused.lance", num_shards=10_001)


def test_load_lance_caps_automatic_shards_without_dropping_fragments() -> None:
    source = object.__new__(LanceSource)
    source.num_shards = None
    source.max_rows = None
    source._planned_rows_by_fragment = None
    source._dataset_cache = type(
        "Dataset",
        (),
        {"get_fragments": lambda self: [None] * 1_001},
    )()

    shards = source.list_shards()

    assert len(shards) == 1_000
    assert shards[0].descriptor.start == 0
    assert shards[-1].descriptor.end == 1_001


def test_load_lance_pins_version_and_shards_by_fragment(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "input.lance"
    version_one = lance.write_dataset(
        pa.table({"x": [1, 2, 3]}),
        str(dataset_uri),
        max_rows_per_file=2,
    ).version
    lance.write_dataset(
        pa.table({"x": [4]}),
        str(dataset_uri),
        mode="append",
    )

    pipeline = load_lance(
        dataset_uri,
        version=version_one,
        columns=["x"],
        batch_size=1,
    )
    shards = pipeline.list_shards()

    assert isinstance(pipeline.source, LanceSource)
    assert pipeline.source.version == version_one
    assert len(shards) == 2
    assert all(isinstance(shard.descriptor, RowRangeDescriptor) for shard in shards)
    assert [(shard.descriptor.start, shard.descriptor.end) for shard in shards] == [
        (0, 1),
        (1, 2),
    ]
    assert [int(row["x"]) for row in pipeline.iter_rows()] == [1, 2, 3]


def test_load_lance_max_rows_bounds_fragment_plan_and_final_batch(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "limited.lance"
    lance.write_dataset(
        pa.table({"x": list(range(6))}),
        str(dataset_uri),
        max_rows_per_file=2,
    )

    pipeline = load_lance(dataset_uri, batch_size=2, max_rows=3)
    shards = pipeline.list_shards()

    assert [(shard.descriptor.start, shard.descriptor.end) for shard in shards] == [
        (0, 1),
        (1, 2),
    ]
    assert [int(row["x"]) for row in pipeline.iter_rows()] == [0, 1, 2]
    assert pipeline.source.describe()["max_rows"] == 3


def test_load_lance_max_rows_zero_and_negative(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "zero.lance"
    lance.write_dataset(pa.table({"x": [1]}), str(dataset_uri))

    assert load_lance(dataset_uri, max_rows=0).list_shards() == []
    assert list(load_lance(dataset_uri, max_rows=0).iter_rows()) == []
    with pytest.raises(ValueError, match="max_rows must be >= 0"):
        load_lance(dataset_uri, max_rows=-1)


def test_load_lance_max_rows_groups_only_required_fragments(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "grouped-limited.lance"
    lance.write_dataset(
        pa.table({"x": list(range(6))}),
        str(dataset_uri),
        max_rows_per_file=2,
    )

    pipeline = load_lance(dataset_uri, max_rows=3, num_shards=1)
    shards = pipeline.list_shards()

    assert len(shards) == 1
    assert (shards[0].descriptor.start, shards[0].descriptor.end) == (0, 2)
    assert [int(row["x"]) for row in pipeline.iter_rows()] == [0, 1, 2]


def test_load_lance_normalizes_classic_blobs_to_references(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "blobs.lance"
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field(
                "image",
                pa.large_binary(),
                metadata={b"lance-encoding:blob": b"true"},
            ),
        ]
    )
    lance.write_dataset(
        pa.Table.from_arrays(
            [
                pa.array([1, 2, 3]),
                pa.array([b"first", None, b"second"], type=pa.large_binary()),
            ],
            schema=schema,
        ),
        str(dataset_uri),
        max_rows_per_file=1,
    )

    pipeline = load_lance(dataset_uri, batch_size=1)
    output_schema = pipeline.output_schema()
    rows = list(pipeline.iter_rows())

    assert output_schema is not None
    assert datatype.asset_storage(output_schema.field("image")) == "blob_reference"
    assert read_blob(rows[0]["image"]) == b"first"
    assert rows[1]["image"] is None
    assert read_blob(rows[2]["image"]) == b"second"
    assert rows[0]["image"]["path"].startswith(f"{dataset_uri}/data/")
    assert rows[0]["image"]["offset"] == 0
    assert rows[0]["image"]["size"] == 5


def test_load_lance_rejects_selected_blob_v2_columns(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    if not hasattr(lance, "blob_array"):
        pytest.skip("Lance Blob V2 is unavailable")
    dataset_uri = tmp_path / "blobs-v2.lance"
    lance.write_dataset(
        pa.table({"id": [1], "image": lance.blob_array([b"image"])}),
        str(dataset_uri),
        data_storage_version="2.2",
    )

    with pytest.raises(ValueError, match="Lance Blob V2 columns"):
        load_lance(dataset_uri)

    assert [
        row["id"] for row in load_lance(dataset_uri, columns=["id"]).iter_rows()
    ] == [1]


def test_load_lance_groups_fragments_into_requested_shards(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "grouped.lance"
    dataset = lance.write_dataset(
        pa.table({"x": list(range(5))}),
        str(dataset_uri),
        max_rows_per_file=1,
    )
    fragment_count = len(dataset.get_fragments())

    pipeline = load_lance(dataset_uri, num_shards=2)
    shards = pipeline.list_shards()

    assert len(shards) == 2
    assert [shard.descriptor.end - shard.descriptor.start for shard in shards] == [
        (fragment_count + 1) // 2,
        fragment_count // 2,
    ]
    assert [int(row["x"]) for row in pipeline.iter_rows()] == list(range(5))
    assert len(load_lance(dataset_uri, num_shards=100).list_shards()) == fragment_count


def test_load_lance_limits_leading_rows_across_fragments(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "limited.lance"
    lance.write_dataset(
        pa.table({"x": list(range(10))}),
        str(dataset_uri),
        max_rows_per_file=3,
    )

    pipeline = load_lance(
        dataset_uri,
        batch_size=2,
        num_shards=2,
        max_rows=5,
    )

    assert [int(row["x"]) for row in pipeline.iter_rows()] == list(range(5))
    assert len(pipeline.list_shards()) == 2
    assert pipeline.source.describe()["max_rows"] == 5


def test_load_lance_max_rows_allows_short_dataset(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "short.lance"
    lance.write_dataset(pa.table({"x": [1, 2]}), str(dataset_uri))

    assert [
        int(row["x"]) for row in load_lance(dataset_uri, max_rows=10).iter_rows()
    ] == [1, 2]


def test_limited_lance_source_rejects_add_columns(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "limited-add-columns.lance"
    lance.write_dataset(pa.table({"x": [1, 2]}), str(dataset_uri))

    with pytest.raises(ValueError, match="limited Lance source"):
        load_lance(dataset_uri, max_rows=1).write_lance_dataset(
            dataset_uri,
            mode="add_columns",
            columns=["y"],
        )


def test_limited_lance_source_adds_columns_with_null_fill(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "limited-add-columns-fill.lance"
    base = lance.write_dataset(
        pa.table({"x": list(range(8))}),
        str(dataset_uri),
        max_rows_per_file=3,
    )

    (
        load_lance(dataset_uri, version=base.version, max_rows=4, batch_size=2)
        .map(lambda row: {"y": int(row["x"]) * 10}, dtypes={"y": datatype.int64()})
        .write_lance_dataset(
            dataset_uri,
            mode=AddColumns(),
            columns=["y"],
        )
        .launch_local(
            name="limited-add-columns-fill",
            num_workers=1,
            rundir=str(tmp_path / "limited-add-columns-fill-run"),
        )
    )

    assert lance.dataset(str(dataset_uri)).to_table().to_pydict() == {
        "x": list(range(8)),
        "y": [0, 10, 20, 30, None, None, None, None],
    }


def test_load_lance_uses_physical_row_addresses_as_source_row_ids(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "positions.lance"
    lance.write_dataset(
        pa.table({"x": [1, 2, 3]}),
        str(dataset_uri),
        max_rows_per_file=2,
    )

    pipeline = load_lance(dataset_uri, batch_size=1)
    addresses_by_shard = []
    for shard in pipeline.list_shards():
        addresses = []
        for unit in pipeline.source.read_shard(shard):
            assert isinstance(unit, Tabular)
            addresses.extend(
                int(value.as_py()) for value in unit.table.column(SOURCE_ROW_ID_COLUMN)
            )
            assert unit.table.column_names == ["x", SOURCE_ROW_ID_COLUMN]
        addresses_by_shard.append(addresses)

    assert addresses_by_shard == [[0, 1], [1 << 32]]


def test_load_lance_hides_internal_columns_from_iter_rows(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "public-rows.lance"
    lance.write_dataset(pa.table({"x": [1]}), str(dataset_uri))

    row = next(iter(load_lance(dataset_uri).iter_rows()))
    output_schema = load_lance(dataset_uri).output_schema()

    assert row.to_dict() == {"x": 1}
    assert output_schema is not None
    assert output_schema.names == ["x"]


def test_load_lance_rejects_configured_fsspec_handle() -> None:
    with pytest.raises(ValueError, match="configured fsspec handles"):
        load_lance(("bucket/data.lance", MemoryFileSystem()))


def test_load_lance_rejects_configured_fsspec_setter() -> None:
    from refiner.io.datafolder import DataFolder

    input_folder = DataFolder("bucket/data.lance")
    input_folder.fs = MemoryFileSystem()

    with pytest.raises(ValueError, match="configured fsspec handles"):
        load_lance(input_folder)


@pytest.mark.parametrize(
    "uri",
    [
        "s3://user:password@bucket/dataset.lance",
        "s3://bucket/data.lance?token=x",
        "s3://bucket/data.lance#token=x",
    ],
)
def test_load_lance_rejects_secret_bearing_uri(uri: str) -> None:
    with pytest.raises(ValueError, match="must not contain credentials"):
        load_lance(uri)


def test_load_lance_rejects_reserved_shard_id_column(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "reserved-shard-id.lance"
    lance.write_dataset(
        pa.table({"__shard_id": ["user-value"], "x": [1]}), str(dataset_uri)
    )

    with pytest.raises(ValueError, match="reserved column __shard_id"):
        load_lance(dataset_uri)


def test_load_lance_cache_does_not_cross_source_instances(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "recreated.lance"
    old_uri = tmp_path / "old-recreated.lance"
    lance.write_dataset(pa.table({"x": [1]}), str(dataset_uri))
    first = load_lance(dataset_uri)
    assert [int(row["x"]) for row in first.iter_rows()] == [1]

    dataset_uri.rename(old_uri)
    lance.write_dataset(pa.table({"x": [2]}), str(dataset_uri))
    second = load_lance(dataset_uri)

    assert [int(row["x"]) for row in second.iter_rows()] == [2]
