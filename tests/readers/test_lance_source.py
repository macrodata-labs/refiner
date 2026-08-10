from __future__ import annotations

import pyarrow as pa
import pytest
from fsspec.implementations.memory import MemoryFileSystem

from refiner import load_lance
from refiner.pipeline.data.shard import RowRangeDescriptor
from refiner.pipeline.sources.lance import (
    LANCE_FRAGMENT_ID_COLUMN,
    LANCE_ROW_POSITION_COLUMN,
)
from refiner.pipeline.sources.lance import LanceSource
from refiner.pipeline.data.tabular import Tabular


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


def test_load_lance_emits_fragment_ids_and_local_row_positions(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "positions.lance"
    lance.write_dataset(
        pa.table({"x": [1, 2, 3]}),
        str(dataset_uri),
        max_rows_per_file=2,
    )

    pipeline = load_lance(dataset_uri, batch_size=1)
    positions_by_shard = []
    fragment_ids_by_shard = []
    for shard in pipeline.list_shards():
        positions = []
        fragment_ids = []
        for unit in pipeline.source.read_shard(shard):
            assert isinstance(unit, Tabular)
            fragment_ids.extend(
                int(value.as_py())
                for chunk in unit.table[LANCE_FRAGMENT_ID_COLUMN].chunks
                for value in chunk
            )
            positions.extend(
                int(value.as_py())
                for chunk in unit.table[LANCE_ROW_POSITION_COLUMN].chunks
                for value in chunk
            )
        positions_by_shard.append(positions)
        fragment_ids_by_shard.append(fragment_ids)

    assert positions_by_shard == [[0, 1], [0]]
    assert len(set(fragment_ids_by_shard[0])) == 1
    assert len(set(fragment_ids_by_shard[1])) == 1
    assert fragment_ids_by_shard[0][0] != fragment_ids_by_shard[1][0]


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
