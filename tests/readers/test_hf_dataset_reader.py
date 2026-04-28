from __future__ import annotations

import sys
import types

import pyarrow as pa
import pytest

from refiner.pipeline.data import datatype
from refiner.pipeline.data.shard import FilePart, Shard
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.expressions import col
from refiner.pipeline.sources.readers import hf_dataset
from refiner.pipeline.sources.readers.hf_dataset import HFDatasetReader


def _feature(type_name: str) -> object:
    return type(type_name, (), {})()


def _parquet_shard(end: int = -1) -> Shard:
    return Shard.from_file_parts(
        [FilePart(path="source.parquet", start=0, end=end)],
        global_ordinal=0,
    )


def _install_datasets(
    monkeypatch,
    *,
    config: str = "default",
    features: dict[str, object] | None = None,
    dataset: object | None = None,
) -> list[tuple[str, str, dict[str, object]]]:
    calls: list[tuple[str, str, dict[str, object]]] = []
    module = types.ModuleType("datasets")

    def get_dataset_config_info(repo: str, **kwargs: object) -> object:
        calls.append(("info", repo, kwargs))
        return types.SimpleNamespace(config_name=config, features=features or {})

    def load_dataset(repo: str, config_name: str, **kwargs: object) -> object:
        calls.append(("load", repo, {"config": config_name, **kwargs}))
        assert dataset is not None
        return dataset

    setattr(module, "get_dataset_config_info", get_dataset_config_info)
    setattr(module, "load_dataset", load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", module)
    return calls


def test_resolve_hf_filepaths_rewrites_only_relative_file_values() -> None:
    table = datatype.apply_dtypes_to_table(
        pa.table(
            {
                "frames": [
                    "relative/path.mp4",
                    "./nested/path.mp4",
                    "https://example.com/path.mp4",
                    "hf://datasets/repo/path.mp4",
                    "s3://bucket/path.mp4",
                    "gs://bucket/path.mp4",
                    "memory://bucket/path.mp4",
                    "az+https://account/path.mp4",
                    "/local/path.mp4",
                    None,
                ],
                "text": ["unchanged"] * 10,
            }
        ),
        {"frames": datatype.video_file()},
    )

    out = hf_dataset.resolve_hf_filepaths(table, "org/repo")

    assert out.column("frames").to_pylist() == [
        "hf://datasets/org/repo/relative/path.mp4",
        "hf://datasets/org/repo/nested/path.mp4",
        "https://example.com/path.mp4",
        "hf://datasets/repo/path.mp4",
        "s3://bucket/path.mp4",
        "gs://bucket/path.mp4",
        "memory://bucket/path.mp4",
        "az+https://account/path.mp4",
        "/local/path.mp4",
        None,
    ]
    assert out.column("text").to_pylist() == ["unchanged"] * 10
    assert out.schema.field("frames").metadata == {b"asset_type": b"video"}


def test_hf_dataset_reader_lists_parquet_and_resolves_file_dtypes(monkeypatch) -> None:
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
        if url == "https://huggingface.co/api/datasets/org/repo/parquet":
            return {
                "cfg": {
                    "test": ["https://huggingface.co/datasets/org/repo/test.parquet"],
                    "train": [
                        "https://huggingface.co/datasets/org/repo/resolve/refs%2Fconvert%2Fparquet/cfg/train/0.parquet"
                    ],
                }
            }
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
                    "image": pa.array(
                        [
                            {"bytes": None, "path": "image.png"},
                            {"bytes": None, "path": None},
                        ],
                        type=pa.struct(
                            [
                                pa.field("bytes", pa.binary()),
                                pa.field("path", pa.string()),
                            ]
                        ),
                    ),
                    "audio": ["sound.wav", "hf://datasets/org/repo/sound.wav"],
                }
            )
            yield Tabular(table)

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)
    monkeypatch.setattr(hf_dataset, "ParquetReader", FakeParquetReader)

    reader = HFDatasetReader(
        "org/repo",
        config="cfg",
        split="train",
        dtypes={
            "frames": datatype.video_file(),
            "image": datatype.image_file(),
            "audio": datatype.audio_file(),
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
    assert parquet_dtypes[0] is None
    assert parquet_filters[0] is None
    units = list(reader.read_shard(shard))

    assert calls
    assert isinstance(units[0], Tabular)
    table = units[0].table
    assert table.num_rows == 1
    assert table.column("frames").to_pylist() == [
        "hf://datasets/org/repo/clip.mp4",
    ]
    assert table.column("image").to_pylist() == [
        "hf://datasets/org/repo/image.png",
    ]
    assert table.column("audio").to_pylist() == [
        "hf://datasets/org/repo/sound.wav",
    ]
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
        if url == "https://huggingface.co/api/datasets/org/repo/parquet":
            return {"default": {"train": ["https://example.com/train.parquet"]}}
        raise AssertionError(url)

    class FakeParquetReader:
        def __init__(self, inputs, **kwargs):
            self.inputs = inputs
            self.dtypes = kwargs["dtypes"]

        def list_shards(self):
            return [_parquet_shard()]

        def read_shard(self, shard):
            del shard
            yield Tabular(
                pa.table(
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
                        "audio": ["sound.wav"],
                        "label": ["cat"],
                    }
                )
            )

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)
    monkeypatch.setattr(hf_dataset, "ParquetReader", FakeParquetReader)

    reader = HFDatasetReader("org/repo", resolve_filepaths=False)

    schema = reader.schema
    assert schema is not None
    assert schema.names == ["image", "audio"]
    assert schema.field("image").metadata == {b"asset_type": b"image"}
    assert schema.field("audio").metadata == {b"asset_type": b"audio"}
    units = list(reader.read_shard(reader.list_shards()[0]))
    assert isinstance(units[0], Tabular)
    table = units[0].table
    assert table.column("image").to_pylist() == ["image.png"]
    assert table.column("audio").to_pylist() == ["sound.wav"]
    assert table.column("label").to_pylist() == ["cat"]
    assert table.schema.field("image").metadata == {b"asset_type": b"image"}
    assert table.schema.field("audio").metadata == {b"asset_type": b"audio"}
    assert calls == ["https://huggingface.co/api/datasets/org/repo/parquet"]


def test_hf_dataset_reader_falls_back_to_repo_parquet_files(monkeypatch) -> None:
    _install_datasets(monkeypatch, config="cfg")
    delegate_inputs: list[object] = []

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        if url == "https://huggingface.co/api/datasets/org/repo/parquet":
            return {}
        if url == "https://huggingface.co/api/datasets/org/repo/parquet/cfg/train":
            return []
        if (
            url
            == "https://huggingface.co/api/datasets/org/repo/parquet/cfg/partial-train"
        ):
            return []
        if "datasets-server.huggingface.co/parquet" in url:
            return {"parquet_files": []}
        if "huggingface.co/api/datasets/org/repo/tree/main/cfg" in url:
            return [
                {
                    "type": "file",
                    "path": "cfg/train-00000-of-00001.parquet",
                }
            ]
        if (
            "huggingface.co/api/datasets/org/repo/tree/refs%2Fconvert%2Fparquet/cfg/train"
            in url
        ):
            raise AssertionError("convert tree should not be needed")
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
        [
            "https://huggingface.co/datasets/org/repo/resolve/main/cfg/train-00000-of-00001.parquet"
        ]
    ]


def test_hf_dataset_reader_falls_back_to_convert_parquet_tree(monkeypatch) -> None:
    _install_datasets(monkeypatch, config="cfg")
    delegate_inputs: list[object] = []

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        if url == "https://huggingface.co/api/datasets/org/repo/parquet":
            return {}
        if url.startswith("https://huggingface.co/api/datasets/org/repo/parquet/cfg/"):
            return []
        if "datasets-server.huggingface.co/parquet" in url:
            return {"parquet_files": []}
        if "huggingface.co/api/datasets/org/repo/tree/main/cfg" in url:
            return []
        if (
            "huggingface.co/api/datasets/org/repo/tree/refs%2Fconvert%2Fparquet/cfg/train"
            in url
        ):
            return [
                {
                    "type": "file",
                    "path": "cfg/train/0000.parquet",
                }
            ]
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
        [
            "https://huggingface.co/datasets/org/repo/resolve/refs%2Fconvert%2Fparquet/cfg/train/0000.parquet"
        ]
    ]


def test_hf_dataset_reader_leaves_bytes_only_media_feature_raw(monkeypatch) -> None:
    _install_datasets(monkeypatch, features={"image": _feature("Image")})

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        if url == "https://huggingface.co/api/datasets/org/repo/parquet":
            return {"default": {"train": ["https://example.com/train.parquet"]}}
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


def test_hf_file_filter_loads_filter_column_before_final_projection(
    monkeypatch,
) -> None:
    _install_datasets(monkeypatch)
    delegate_columns: list[tuple[str, ...] | None] = []

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        if url == "https://huggingface.co/api/datasets/org/repo/parquet":
            return {"default": {"train": ["https://example.com/train.parquet"]}}
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
                        "audio": ["a.wav", "b.wav"],
                        "image": pa.array(
                            [
                                {"bytes": None, "path": "keep.png"},
                                {"bytes": None, "path": "drop.jpg"},
                            ],
                            type=pa.struct(
                                [
                                    pa.field("bytes", pa.binary()),
                                    pa.field("path", pa.string()),
                                ]
                            ),
                        ),
                        "file_path": ["source.parquet", "source.parquet"],
                    }
                )
            )

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)
    monkeypatch.setattr(hf_dataset, "ParquetReader", FakeParquetReader)

    reader = HFDatasetReader(
        "org/repo",
        columns_to_read=("audio",),
        dtypes={
            "audio": datatype.audio_file(),
            "image": datatype.image_file(),
        },
        filter=col("image") == "keep.png",
    )

    units = list(reader.read_shard(reader.list_shards()[0]))

    assert delegate_columns == [("audio", "image")]
    assert isinstance(units[0], Tabular)
    table = units[0].table
    assert table.column_names == ["audio", "file_path"]
    assert table.column("audio").to_pylist() == ["hf://datasets/org/repo/a.wav"]
    assert table.column("file_path").to_pylist() == ["source.parquet"]


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
                    "video": ["drop.mp4", "keep.mp4"],
                }
            )

    calls = _install_datasets(
        monkeypatch,
        features={"video": _feature("Video")},
        dataset=FakeStreamingDataset(),
    )

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        if url == "https://huggingface.co/api/datasets/org/repo/parquet":
            return {}
        if url.startswith("https://huggingface.co/api/datasets/org/repo/parquet/"):
            return []
        if "datasets-server.huggingface.co/parquet" in url:
            return {"parquet_files": []}
        if "/tree/" in url:
            return []
        raise AssertionError(url)

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)

    reader = HFDatasetReader(
        "org/repo",
        num_shards=2,
        arrow_batch_size=10,
        dtypes={"id": datatype.int64()},
        filter=col("label") == "keep",
    )
    shards = reader.list_shards()
    units = list(reader.read_shard(shards[1]))

    assert [call[0] for call in calls] == ["info", "load"]
    assert len(shards) == 2
    assert isinstance(units[0], Tabular)
    table = units[0].table
    assert table.column("id").to_pylist() == [11]
    assert table.column("id").type == pa.int64()
    assert table.column("label").to_pylist() == ["keep"]
    assert table.column("video").to_pylist() == ["hf://datasets/org/repo/keep.mp4"]
    assert table.schema.field("video").metadata == {b"asset_type": b"video"}


def test_hf_dataset_fallback_errors_when_requested_shards_exceed_source_shards(
    monkeypatch,
) -> None:
    class FakeStreamingDataset:
        num_shards = 1

    _install_datasets(monkeypatch, dataset=FakeStreamingDataset())

    def fake_get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
        del hf_token, timeout
        if url == "https://huggingface.co/api/datasets/org/repo/parquet":
            return {}
        if url.startswith("https://huggingface.co/api/datasets/org/repo/parquet/"):
            return []
        if "datasets-server.huggingface.co/parquet" in url:
            return {"parquet_files": []}
        if "/tree/" in url:
            return []
        raise AssertionError(url)

    monkeypatch.setattr(hf_dataset, "_get_json", fake_get_json)

    reader = HFDatasetReader("org/repo", num_shards=2)

    with pytest.raises(ValueError, match="only 1 source shard"):
        reader.list_shards()
