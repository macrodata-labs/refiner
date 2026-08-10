from __future__ import annotations

import pyarrow as pa
import pytest

from refiner import load_lance
from refiner.pipeline.data.shard import LanceFragmentDescriptor, Shard
from refiner.pipeline.sources.lance import LANCE_ROW_POSITION_COLUMN
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
    assert all(
        isinstance(shard.descriptor, LanceFragmentDescriptor) for shard in shards
    )
    assert sum(shard.descriptor.num_rows for shard in shards) == 3
    assert [int(row["x"]) for row in pipeline.iter_rows()] == [1, 2, 3]


def test_lance_fragment_descriptor_roundtrips() -> None:
    shard = Shard.from_lance_fragment(
        dataset_uri="s3://bucket/data.lance",
        version=42,
        fragment_id=7,
        num_rows=123,
        global_ordinal=2,
    )

    restored = Shard.from_dict(shard.to_dict())

    assert restored == shard
    assert isinstance(restored.descriptor, LanceFragmentDescriptor)


def test_load_lance_emits_fragment_local_row_positions(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "positions.lance"
    lance.write_dataset(
        pa.table({"x": [1, 2, 3]}),
        str(dataset_uri),
        max_rows_per_file=2,
    )

    pipeline = load_lance(dataset_uri, batch_size=1)
    positions_by_shard = []
    for shard in pipeline.list_shards():
        positions = []
        for unit in pipeline.source.read_shard(shard):
            assert isinstance(unit, Tabular)
            positions.extend(
                int(value.as_py())
                for chunk in unit.table[LANCE_ROW_POSITION_COLUMN].chunks
                for value in chunk
            )
        positions_by_shard.append(positions)

    assert positions_by_shard == [[0, 1], [0]]
