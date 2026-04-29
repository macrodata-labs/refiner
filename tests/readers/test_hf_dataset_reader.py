from __future__ import annotations

import types

import pyarrow as pa
import pytest
import datasets
from datasets.packaged_modules.parquet.parquet import Parquet

from refiner.pipeline.data import datatype
from refiner.pipeline.data.shard import FilePart, FilePartsDescriptor, Shard
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.expressions import col
from refiner.pipeline.sources.readers import hf_dataset
from refiner.pipeline.sources.readers.hf_dataset import HFDatasetReader


def _feature(type_name: str) -> object:
    return type(type_name, (), {})()


def _parquet_shard() -> Shard:
    return Shard.from_file_parts(
        [FilePart(path="source.parquet", start=0, end=-1)],
        global_ordinal=0,
    )


def _empty_parquet_resolution(url: str) -> object:
    if url.startswith("https://huggingface.co/api/datasets/org/repo/parquet/"):
        return []
    if "datasets-server.huggingface.co/parquet" in url:
        return {"parquet_files": []}
    raise AssertionError(url)


def _install_datasets(
    monkeypatch,
    *,
    config: str = "default",
    features: dict[str, object] | None = None,
    dataset: object | None = None,
    builder: object | None = None,
) -> list[tuple[str, str, dict[str, object]]]:
    calls: list[tuple[str, str, dict[str, object]]] = []

    def get_dataset_config_info(repo: str, **kwargs: object) -> object:
        calls.append(("info", repo, kwargs))
        return types.SimpleNamespace(config_name=config, features=features or {})

    def load_dataset_builder(repo: str, config_name: str, **kwargs: object) -> object:
        calls.append(("builder", repo, {"config": config_name, **kwargs}))
        return builder or types.SimpleNamespace(config=types.SimpleNamespace())

    def load_dataset(repo: str, config_name: str, **kwargs: object) -> object:
        calls.append(("load", repo, {"config": config_name, **kwargs}))
        assert dataset is not None
        return dataset

    monkeypatch.setattr(datasets, "get_dataset_config_info", get_dataset_config_info)
    monkeypatch.setattr(datasets, "load_dataset_builder", load_dataset_builder)
    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    return calls


def _parquet_builder(data_files: object) -> object:
    builder = object.__new__(Parquet)
    builder.config = types.SimpleNamespace(data_files=data_files)
    return builder


def test_hf_dataset_reader_lists_parquet_and_preserves_asset_path_values(
    monkeypatch,
) -> None:
    _install_datasets(monkeypatch, config="cfg")
    calls: list[str] = []
    storage_options: list[object] = []
    parquet_dtypes: list[object] = []
    parquet_filters: list[object] = []
    parquet_inputs: list[object] = []

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        calls.append(url)
        assert hf_token == "tok"
        assert timeout == 5.0
        if url == "https://huggingface.co/api/datasets/org/repo/parquet/cfg/train":
            return [
                "https://huggingface.co/datasets/org/repo/resolve/refs%2Fconvert%2Fparquet/cfg/train/0.parquet"
            ]
        raise AssertionError(url)

    class FakeParquetReader:
        def __init__(self, inputs, **kwargs):
            self.inputs = inputs
            parquet_inputs.append(inputs)
            self.dtypes = kwargs["dtypes"]
            self.filter = kwargs["filter"]
            self.storage_options = kwargs["storage_options"]
            storage_options.append(self.storage_options)
            parquet_dtypes.append(self.dtypes)
            parquet_filters.append(self.filter)

        def list_shards(self):
            return [_parquet_shard()]

        def read_shard(self, shard):
            del shard
            table = pa.table(
                {
                    "frames": ["clip.mp4", "https://example.com/clip.mp4"],
                    "image": ["image.png", None],
                    "audio": ["sound.wav", "hf://datasets/org/repo/sound.wav"],
                }
            )
            table = datatype.apply_dtypes_to_table(table, self.dtypes, strict=False)
            if self.filter is not None:
                table = hf_dataset.filter_table(table, self.filter)
            yield Tabular(table)

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)
    monkeypatch.setattr(hf_dataset, "ParquetReader", FakeParquetReader)

    reader = HFDatasetReader(
        "org/repo",
        config="cfg",
        split="train",
        dtypes={
            "frames": datatype.video_path(),
            "image": datatype.image_path(),
            "audio": datatype.audio_path(),
        },
        filter=col("image") == "image.png",
        hf_token="tok",
        timeout=5.0,
    )
    schema = reader.schema
    assert schema is not None
    assert schema.names == ["frames", "image", "audio"]
    assert schema.field("frames").metadata == {b"asset_type": b"video"}
    assert schema.field("image").metadata == {b"asset_type": b"image"}
    assert schema.field("audio").metadata == {b"asset_type": b"audio"}

    shard = reader.list_shards()[0]
    assert parquet_inputs[0] == [
        "https://huggingface.co/datasets/org/repo/resolve/refs%2Fconvert%2Fparquet/cfg/train/0.parquet"
    ]
    assert storage_options[0] == {"headers": {"Authorization": "Bearer tok"}}
    assert parquet_dtypes[0] == {
        "frames": datatype.video_path(),
        "image": datatype.image_path(),
        "audio": datatype.audio_path(),
    }
    assert parquet_filters[0] is not None
    units = list(reader.read_shard(shard))

    assert calls
    assert isinstance(units[0], Tabular)
    table = units[0].table
    assert table.num_rows == 1
    assert table.column("frames").to_pylist() == ["clip.mp4"]
    assert table.column("image").to_pylist() == ["image.png"]
    assert table.column("audio").to_pylist() == ["sound.wav"]
    assert table.schema.field("frames").metadata == {b"asset_type": b"video"}
    assert table.schema.field("image").metadata == {b"asset_type": b"image"}
    assert table.schema.field("audio").metadata == {b"asset_type": b"audio"}


def test_hf_dataset_reader_infers_media_dtypes_from_features(monkeypatch) -> None:
    _install_datasets(
        monkeypatch,
        features={
            "image": _feature("Image"),
            "audio": _feature("Audio"),
            "label": _feature("Value"),
        },
    )
    calls: list[str] = []

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        calls.append(url)
        if url == "https://huggingface.co/api/datasets/org/repo/parquet/default/train":
            return ["https://example.com/train.parquet"]
        raise AssertionError(url)

    class FakeParquetReader:
        def __init__(self, inputs, **kwargs):
            self.inputs = inputs
            self.dtypes = kwargs["dtypes"]

        def list_shards(self):
            return [_parquet_shard()]

        def read_shard(self, shard):
            del shard
            table = pa.table(
                {
                    "image": pa.array(
                        [{"bytes": None, "path": "image.png"}],
                        type=pa.struct(
                            [
                                pa.field("bytes", pa.binary()),
                                pa.field("path", pa.string()),
                            ]
                        ),
                    ),
                    "audio": pa.array(
                        [{"bytes": None, "path": "sound.wav"}],
                        type=pa.struct(
                            [
                                pa.field("bytes", pa.binary()),
                                pa.field("path", pa.string()),
                            ]
                        ),
                    ),
                    "label": ["cat"],
                }
            )
            yield Tabular(
                datatype.apply_dtypes_to_table(
                    table,
                    self.dtypes,
                    strict=False,
                )
            )

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)
    monkeypatch.setattr(hf_dataset, "ParquetReader", FakeParquetReader)

    reader = HFDatasetReader("org/repo")

    schema = reader.schema
    assert schema is not None
    assert schema.names == ["image", "audio"]
    assert schema.field("image").metadata == {b"asset_type": b"image"}
    assert schema.field("audio").metadata == {b"asset_type": b"audio"}
    units = list(reader.read_shard(reader.list_shards()[0]))
    assert isinstance(units[0], Tabular)
    table = units[0].table
    assert table.column("image").to_pylist() == [{"bytes": None, "path": "image.png"}]
    assert table.column("audio").to_pylist() == [{"bytes": None, "path": "sound.wav"}]
    assert table.column("label").to_pylist() == ["cat"]
    assert table.schema.field("image").metadata == {b"asset_type": b"image"}
    assert table.schema.field("audio").metadata == {b"asset_type": b"audio"}
    assert datatype.asset_storage(table.schema.field("image")) == "bytes_with_path"
    assert datatype.asset_storage(table.schema.field("audio")) == "bytes_with_path"
    assert calls == [
        "https://huggingface.co/api/datasets/org/repo/parquet/default/train"
    ]


def test_hf_dataset_reader_uses_datasets_server_parquet_files(monkeypatch) -> None:
    _install_datasets(monkeypatch, config="cfg")
    delegate_inputs: list[object] = []

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        if url == "https://huggingface.co/api/datasets/org/repo/parquet/cfg/train":
            return []
        if "datasets-server.huggingface.co/parquet" in url:
            return {
                "parquet_files": [
                    {
                        "config": "cfg",
                        "split": "train",
                        "url": "https://example.com/cfg/train.parquet",
                    },
                ]
            }
        raise AssertionError(url)

    class FakeParquetReader:
        def __init__(self, inputs, **kwargs):
            del kwargs
            delegate_inputs.append(inputs)

        def list_shards(self):
            return [_parquet_shard()]

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)
    monkeypatch.setattr(hf_dataset, "ParquetReader", FakeParquetReader)

    reader = HFDatasetReader("org/repo", config="cfg")

    assert reader.list_shards()
    assert delegate_inputs == [["https://example.com/cfg/train.parquet"]]


def test_hf_dataset_reader_uses_parquet_builder_data_files(monkeypatch) -> None:
    builder = _parquet_builder(
        {
            "train": [
                "hf://datasets/org/repo@abc/cfg/train-00000-of-00001.parquet",
                "hf://datasets/org/repo@abc/cfg/sidecar.json",
            ],
        }
    )
    _install_datasets(monkeypatch, config="cfg", builder=builder)
    delegate_inputs: list[object] = []

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        if url == "https://huggingface.co/api/datasets/org/repo/parquet/cfg/train":
            return []
        if "datasets-server.huggingface.co/parquet" in url:
            return {"parquet_files": []}
        raise AssertionError(url)

    class FakeParquetReader:
        def __init__(self, inputs, **kwargs):
            del kwargs
            delegate_inputs.append(inputs)

        def list_shards(self):
            return [_parquet_shard()]

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)
    monkeypatch.setattr(hf_dataset, "ParquetReader", FakeParquetReader)

    reader = HFDatasetReader("org/repo", config="cfg")

    assert reader.list_shards()
    assert delegate_inputs == [
        ["hf://datasets/org/repo@abc/cfg/train-00000-of-00001.parquet"]
    ]


def test_hf_dataset_reader_leaves_bytes_only_media_feature_raw(monkeypatch) -> None:
    _install_datasets(monkeypatch, features={"image": _feature("Image")})

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        if url == "https://huggingface.co/api/datasets/org/repo/parquet/default/train":
            return ["https://example.com/train.parquet"]
        raise AssertionError(url)

    class FakeParquetReader:
        def __init__(self, inputs, **kwargs):
            del inputs, kwargs

        def list_shards(self):
            return [_parquet_shard()]

        def read_shard(self, shard):
            del shard
            yield Tabular(
                pa.table(
                    {
                        "image": pa.array(
                            [{"bytes": b"encoded"}],
                            type=pa.struct([pa.field("bytes", pa.binary())]),
                        )
                    }
                )
            )

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)
    monkeypatch.setattr(hf_dataset, "ParquetReader", FakeParquetReader)

    reader = HFDatasetReader("org/repo")
    units = list(reader.read_shard(reader.list_shards()[0]))

    assert isinstance(units[0], Tabular)
    table = units[0].table
    assert table.column("image").to_pylist() == [{"bytes": b"encoded"}]
    assert table.schema.field("image").metadata is None


def test_hf_asset_path_filter_delegates_to_parquet_reader(
    monkeypatch,
) -> None:
    _install_datasets(monkeypatch)
    delegate_columns: list[tuple[str, ...] | None] = []
    delegate_filters: list[object] = []

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        if url == "https://huggingface.co/api/datasets/org/repo/parquet/default/train":
            return ["https://example.com/train.parquet"]
        raise AssertionError(url)

    class FakeParquetReader:
        def __init__(self, inputs, **kwargs):
            del inputs
            self.columns_to_read = kwargs["columns_to_read"]
            self.dtypes = kwargs["dtypes"]
            self.filter = kwargs["filter"]
            delegate_columns.append(self.columns_to_read)
            delegate_filters.append(self.filter)

        def list_shards(self):
            return [_parquet_shard()]

        def read_shard(self, shard):
            del shard
            table = pa.table(
                {
                    "audio": ["a.wav", "b.wav"],
                    "image": ["keep.png", "drop.jpg"],
                    "file_path": ["source.parquet", "source.parquet"],
                }
            )
            table = datatype.apply_dtypes_to_table(table, self.dtypes, strict=False)
            if self.filter is not None:
                table = hf_dataset.filter_table(table, self.filter)
            yield Tabular(table)

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)
    monkeypatch.setattr(hf_dataset, "ParquetReader", FakeParquetReader)

    reader = HFDatasetReader(
        "org/repo",
        columns_to_read=("audio",),
        dtypes={
            "audio": datatype.audio_path(),
            "image": datatype.image_path(),
        },
        filter=col("image") == "keep.png",
    )

    units = list(reader.read_shard(reader.list_shards()[0]))

    assert delegate_columns == [("audio",)]
    assert delegate_filters[0] is not None
    assert isinstance(units[0], Tabular)
    table = units[0].table
    assert table.column_names == ["audio", "file_path"]
    assert table.column("audio").to_pylist() == ["a.wav"]
    assert table.column("file_path").to_pylist() == ["source.parquet"]


def test_hf_file_path_filter_runs_after_parquet_reader(monkeypatch) -> None:
    _install_datasets(monkeypatch)
    parquet_filters: list[object] = []

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        if url == "https://huggingface.co/api/datasets/org/repo/parquet/default/train":
            return ["https://example.com/train.parquet"]
        raise AssertionError(url)

    class FakeParquetReader:
        def __init__(self, inputs, **kwargs):
            del inputs
            parquet_filters.append(kwargs["filter"])

        def list_shards(self):
            return [_parquet_shard()]

        def read_shard(self, shard):
            del shard
            yield Tabular(
                pa.table(
                    {
                        "value": [1, 2],
                        "file_path": ["keep.parquet", "drop.parquet"],
                    }
                )
            )

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)
    monkeypatch.setattr(hf_dataset, "ParquetReader", FakeParquetReader)

    reader = HFDatasetReader("org/repo", filter=col("file_path") == "keep.parquet")
    units = list(reader.read_shard(reader.list_shards()[0]))

    assert parquet_filters == [None]
    assert isinstance(units[0], Tabular)
    assert units[0].table.column("value").to_pylist() == [1]


def test_hf_dataset_reader_reads_planned_shard_without_relisting(monkeypatch) -> None:
    _install_datasets(monkeypatch)
    calls: list[str] = []
    delegate_inputs: list[object] = []
    delegate_source_indexes: list[list[int]] = []

    planned_shard = Shard.from_file_parts(
        [
            FilePart(
                path="https://example.com/train-00000.parquet",
                start=0,
                end=10,
                source_index=4,
            ),
            FilePart(
                path="https://example.com/train-00001.parquet",
                start=0,
                end=20,
                source_index=7,
            ),
        ],
        global_ordinal=0,
    )

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        calls.append(url)
        if url == "https://huggingface.co/api/datasets/org/repo/parquet/default/train":
            return ["https://example.com/train-00000.parquet"]
        raise AssertionError(url)

    class FakeParquetReader:
        def __init__(self, inputs, **kwargs):
            del kwargs
            delegate_inputs.append(inputs)

        def list_shards(self):
            return [planned_shard]

        def read_shard(self, shard):
            assert isinstance(shard.descriptor, FilePartsDescriptor)
            delegate_source_indexes.append(
                [part.source_index for part in shard.descriptor.parts]
            )
            yield Tabular(pa.table({"value": [1]}))

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)
    monkeypatch.setattr(hf_dataset, "ParquetReader", FakeParquetReader)

    shard = Shard.from_dict(HFDatasetReader("org/repo").list_shards()[0].to_dict())
    calls.clear()
    units = list(HFDatasetReader("org/repo").read_shard(shard))

    assert calls == []
    assert delegate_inputs[-1] == [
        "https://example.com/train-00000.parquet",
        "https://example.com/train-00001.parquet",
    ]
    assert delegate_source_indexes == [[0, 1]]
    assert isinstance(units[0], Tabular)


def test_hf_dataset_reader_preserves_empty_projection(monkeypatch) -> None:
    _install_datasets(monkeypatch)
    delegate_columns: list[tuple[str, ...] | None] = []

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        if url == "https://huggingface.co/api/datasets/org/repo/parquet/default/train":
            return ["https://example.com/train.parquet"]
        raise AssertionError(url)

    class FakeParquetReader:
        def __init__(self, inputs, **kwargs):
            del inputs
            self.columns_to_read = kwargs["columns_to_read"]
            delegate_columns.append(self.columns_to_read)

        def list_shards(self):
            return [_parquet_shard()]

        def read_shard(self, shard):
            del shard
            yield Tabular(
                pa.table(
                    {
                        "value": [1],
                        "file_path": ["source.parquet"],
                    }
                )
            )

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)
    monkeypatch.setattr(hf_dataset, "ParquetReader", FakeParquetReader)

    reader = HFDatasetReader("org/repo", columns_to_read=[])
    units = list(reader.read_shard(reader.list_shards()[0]))

    assert delegate_columns == [()]
    assert isinstance(units[0], Tabular)
    assert units[0].table.column_names == ["file_path"]


def test_hf_dataset_reader_does_not_mix_fallback_after_parquet_read_error(
    monkeypatch,
) -> None:
    _install_datasets(monkeypatch)

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        if url == "https://huggingface.co/api/datasets/org/repo/parquet/default/train":
            return ["https://example.com/train.parquet"]
        raise AssertionError(url)

    class FakeParquetReader:
        def __init__(self, inputs, **kwargs):
            del inputs, kwargs

        def list_shards(self):
            return [_parquet_shard()]

        def read_shard(self, shard):
            del shard
            raise OSError("bad parquet shard")

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)
    monkeypatch.setattr(hf_dataset, "ParquetReader", FakeParquetReader)

    reader = HFDatasetReader("org/repo")

    with pytest.raises(OSError, match="bad parquet shard"):
        list(reader.read_shard(reader.list_shards()[0]))


def test_hf_dataset_reader_falls_back_to_datasets_streaming(monkeypatch) -> None:
    class FakeStreamingDataset:
        num_shards = 2

        def __init__(self, shard_index: int | None = None):
            self.shard_index = shard_index

        def shard(self, *, num_shards: int, index: int):
            assert num_shards == 2
            return FakeStreamingDataset(index)

        def with_format(self, format_name: str):
            assert format_name == "arrow"
            return self

        def iter(self, *, batch_size: int):
            assert batch_size == 10
            base_id = (self.shard_index or 0) * 10
            yield pa.table(
                {
                    "id": [str(base_id), str(base_id + 1)],
                    "label": ["drop", "keep"],
                    "video": pa.array(
                        [
                            {"bytes": None, "path": "drop.mp4"},
                            {"bytes": None, "path": "keep.mp4"},
                        ],
                        type=pa.struct(
                            [
                                pa.field("bytes", pa.binary()),
                                pa.field("path", pa.string()),
                            ]
                        ),
                    ),
                }
            )

    calls = _install_datasets(
        monkeypatch,
        features={"video": _feature("Video")},
        dataset=FakeStreamingDataset(),
    )

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        return _empty_parquet_resolution(url)

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)

    reader = HFDatasetReader(
        "org/repo",
        num_shards=2,
        arrow_batch_size=10,
        dtypes={"id": datatype.int64()},
        filter=(col("label") == "keep") & col("file_path").is_null(),
    )
    shards = reader.list_shards()
    units = list(reader.read_shard(shards[1]))

    assert [call[0] for call in calls] == ["info", "builder", "load"]
    assert len(shards) == 2
    assert isinstance(units[0], Tabular)
    table = units[0].table
    assert table.column("id").to_pylist() == [11]
    assert table.column("id").type == pa.int64()
    assert table.column("file_path").to_pylist() == [None]
    assert table.column("label").to_pylist() == ["keep"]
    assert table.column("video").to_pylist() == [{"bytes": None, "path": "keep.mp4"}]
    assert table.schema.field("video").metadata == {b"asset_type": b"video"}
    assert datatype.asset_storage(table.schema.field("video")) == "bytes_with_path"


def test_hf_dataset_fallback_errors_when_requested_shards_exceed_source_shards(
    monkeypatch,
) -> None:
    class FakeStreamingDataset:
        num_shards = 1

    _install_datasets(monkeypatch, dataset=FakeStreamingDataset())

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        return _empty_parquet_resolution(url)

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)

    reader = HFDatasetReader("org/repo", num_shards=2)

    with pytest.raises(ValueError, match="only 1 source shard"):
        reader.list_shards()


def test_hf_dataset_fallback_treats_non_positive_shards_as_auto(monkeypatch) -> None:
    class FakeStreamingDataset:
        num_shards = 3

    _install_datasets(monkeypatch, dataset=FakeStreamingDataset())

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        return _empty_parquet_resolution(url)

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)

    reader = HFDatasetReader("org/repo", num_shards=-1)

    assert len(reader.list_shards()) == 3
