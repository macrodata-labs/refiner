from __future__ import annotations

from pathlib import Path

from fsspec.implementations.memory import MemoryFileSystem
import pyarrow as pa
import pytest

from refiner.io import DataFile
from refiner.pipeline import read_tfrecords
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.readers import TfrecordReader

tf = pytest.importorskip("tensorflow")


def _example(row_id: int, text: bytes, values: list[float]) -> bytes:
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[row_id])),
                "text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[text])),
                "values": tf.train.Feature(float_list=tf.train.FloatList(value=values)),
            }
        )
    ).SerializeToString()


def _write_tfrecord(path: Path, rows: list[tuple[int, bytes, list[float]]]) -> None:
    with tf.io.TFRecordWriter(str(path)) as writer:
        for row_id, text, values in rows:
            writer.write(_example(row_id, text, values))


def _rows_from_reader(reader: TfrecordReader) -> list[dict]:
    rows = []
    for shard in reader.list_shards():
        for unit in reader.read_shard(shard):
            assert isinstance(unit, Tabular)
            rows.extend(row.to_dict() for row in unit)
    return rows


def _features():
    return {
        "id": tf.io.FixedLenFeature([], tf.int64),
        "text": tf.io.FixedLenFeature([], tf.string),
        "values": tf.io.FixedLenFeature([2], tf.float32),
    }


def test_tfrecord_reader_reads_multiple_file_shards(tmp_path: Path) -> None:
    first = tmp_path / "first.tfrecord"
    second = tmp_path / "second.tfrecord"
    _write_tfrecord(first, [(1, b"a", [1.0, 1.5])])
    _write_tfrecord(second, [(2, b"b", [2.0, 2.5])])

    reader = TfrecordReader(
        [str(first), str(second)],
        features=_features(),
        target_shard_bytes=1,
        batch_size=1,
    )

    assert len(reader.list_shards()) == 2
    rows = _rows_from_reader(reader)
    assert [row["id"] for row in rows] == [1, 2]
    assert [row["text"] for row in rows] == [b"a", b"b"]
    assert rows[0]["values"] == pytest.approx([1.0, 1.5])
    assert rows[1]["file_path"] == str(second)


def test_tfrecord_reader_keeps_file_path_for_each_source_in_grouped_shard(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.tfrecord"
    second = tmp_path / "second.tfrecord"
    _write_tfrecord(first, [(1, b"a", [1.0, 1.5])])
    _write_tfrecord(second, [(2, b"b", [2.0, 2.5])])

    reader = TfrecordReader(
        [str(first), str(second)],
        features=_features(),
        target_shard_bytes=10**9,
        batch_size=10,
    )

    assert len(reader.list_shards()) == 1
    rows = _rows_from_reader(reader)
    assert [row["file_path"] for row in rows] == [str(first), str(second)]


def test_tfrecord_reader_preserves_fixed_len_feature_dtype(tmp_path: Path) -> None:
    path = tmp_path / "data.tfrecord"
    _write_tfrecord(path, [(1, b"a", [1.0, 1.5])])
    reader = TfrecordReader(str(path), features=_features(), batch_size=4)

    shard = reader.list_shards()[0]
    unit = next(reader.read_shard(shard))

    assert isinstance(unit, Tabular)
    assert unit.unit.schema.field("values").type == pa.list_(pa.float32(), 2)


def test_tfrecord_reader_supports_gzip_auto(tmp_path: Path) -> None:
    path = tmp_path / "data.tfrecord.gz"
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(str(path), options=options) as writer:
        writer.write(_example(3, b"gz", [3.0, 3.5]))

    reader = TfrecordReader(str(path), features=_features(), batch_size=4)

    rows = _rows_from_reader(reader)
    assert rows[0]["id"] == 3
    assert rows[0]["text"] == b"gz"


def test_tfrecord_reader_finds_auto_compressed_directory_files(
    tmp_path: Path,
) -> None:
    gzip_path = tmp_path / "part.tfrec.gz"
    gzip_options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(str(gzip_path), options=gzip_options) as writer:
        writer.write(_example(3, b"gz", [3.0, 3.5]))

    zlib_path = tmp_path / "part.tfrecord.zlib"
    zlib_options = tf.io.TFRecordOptions(compression_type="ZLIB")
    with tf.io.TFRecordWriter(str(zlib_path), options=zlib_options) as writer:
        writer.write(_example(4, b"zlib", [4.0, 4.5]))
    (tmp_path / "notes.gz").write_bytes(b"not a tfrecord")

    reader = TfrecordReader(str(tmp_path), features=_features(), batch_size=4)

    rows = _rows_from_reader(reader)
    assert sorted(row["id"] for row in rows) == [3, 4]


def test_tfrecord_reader_does_not_overwrite_file_path_feature(tmp_path: Path) -> None:
    path = tmp_path / "data.tfrecord"
    with tf.io.TFRecordWriter(str(path)) as writer:
        writer.write(
            tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "id": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[1])
                        ),
                        "file_path": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[b"inside-record"])
                        ),
                    }
                )
            ).SerializeToString()
        )
    reader = TfrecordReader(
        str(path),
        features={
            "id": tf.io.FixedLenFeature([], tf.int64),
            "file_path": tf.io.FixedLenFeature([], tf.string),
        },
    )

    rows = _rows_from_reader(reader)
    assert rows[0]["file_path"] == b"inside-record"


def test_tfrecord_reader_rejects_custom_fsspec_filesystems() -> None:
    fs = MemoryFileSystem()
    fs.pipe("data.tfrecord", b"not-read")
    reader = TfrecordReader(
        DataFile(fs=fs, path="data.tfrecord"),
        features={"id": tf.io.FixedLenFeature([], tf.int64)},
    )

    with pytest.raises(ValueError, match="Custom fsspec filesystems"):
        list(reader.read_shard(reader.list_shards()[0]))


def test_read_tfrecords_pipeline_entrypoint(tmp_path: Path) -> None:
    path = tmp_path / "data.tfrecord"
    _write_tfrecord(path, [(4, b"p", [4.0, 4.5])])

    pipeline = read_tfrecords(str(path), features=_features(), file_path_column=None)

    rows = pipeline.take(1)
    assert rows[0]["id"] == 4
    assert "file_path" not in rows[0]
