from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from refiner.pipeline import read_tfds
from refiner.pipeline.data.shard import RowRangeDescriptor
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.readers import TfdsReader

tfds = pytest.importorskip("tensorflow_datasets")
pytest.importorskip("tensorflow")


class TinyDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "id": tfds.features.Scalar(dtype=np.int64),
                    "text": tfds.features.Text(),
                    "nested": {
                        "value": tfds.features.Scalar(dtype=np.int64),
                    },
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        for idx in range(5):
            yield (
                str(idx),
                {
                    "id": idx,
                    "text": f"row-{idx}",
                    "nested": {"value": idx + 10},
                },
            )


def _rows_from_reader(reader: TfdsReader) -> list[dict]:
    rows = []
    for shard in reader.list_shards():
        for unit in reader.read_shard(shard):
            assert isinstance(unit, Tabular)
            rows.extend(row.to_dict() for row in unit)
    return rows


def test_tfds_reader_shards_by_example_range(tmp_path: Path, monkeypatch) -> None:
    builder = TinyDataset(data_dir=str(tmp_path))
    builder.download_and_prepare()
    monkeypatch.setattr(tfds, "builder", lambda *args, **kwargs: builder)

    reader = TfdsReader("tiny_dataset", examples_per_shard=2, batch_size=2)

    shards = reader.list_shards()
    rows = _rows_from_reader(reader)
    ranges = []
    for shard in shards:
        assert isinstance(shard.descriptor, RowRangeDescriptor)
        ranges.append((shard.descriptor.start, shard.descriptor.end))
    assert ranges == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    assert sorted(row["id"] for row in rows) == [0, 1, 2, 3, 4]
    assert {row["id"]: row["nested"]["value"] for row in rows} == {
        0: 10,
        1: 11,
        2: 12,
        3: 13,
        4: 14,
    }


def test_read_tfds_pipeline_entrypoint(tmp_path: Path, monkeypatch) -> None:
    builder = TinyDataset(data_dir=str(tmp_path))
    builder.download_and_prepare()
    monkeypatch.setattr(tfds, "builder", lambda *args, **kwargs: builder)

    pipeline = read_tfds("tiny_dataset", num_shards=2, batch_size=3)

    assert sorted(row["id"] for row in pipeline.take(5)) == [0, 1, 2, 3, 4]


def test_tfds_reader_requires_plain_split_name(tmp_path: Path, monkeypatch) -> None:
    builder = TinyDataset(data_dir=str(tmp_path))
    builder.download_and_prepare()
    monkeypatch.setattr(tfds, "builder", lambda *args, **kwargs: builder)

    with pytest.raises(ValueError, match="plain split names"):
        TfdsReader("tiny_dataset", split="train[:2]")


def test_tfds_reader_rejects_non_positive_num_shards(
    tmp_path: Path, monkeypatch
) -> None:
    builder = TinyDataset(data_dir=str(tmp_path))
    builder.download_and_prepare()
    monkeypatch.setattr(tfds, "builder", lambda *args, **kwargs: builder)

    with pytest.raises(ValueError, match="num_shards"):
        TfdsReader("tiny_dataset", num_shards=0)
