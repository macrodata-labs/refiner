from __future__ import annotations

import io
from pathlib import Path
from tempfile import TemporaryDirectory

from fsspec.implementations.memory import MemoryFileSystem
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from refiner.io import DataFile, DataFolder
from refiner.io.fileset import DataFileSet
from refiner.pipeline import RefinerPipeline, read_csv, read_json, read_parquet
from refiner.pipeline.expressions import col


def _write_jsonl(path: Path, values: list[int]) -> None:
    path.write_text("".join(f'{{"x": {value}}}\n' for value in values))


def _write_csv(path: Path, values: list[int]) -> None:
    path.write_text("x\n" + "".join(f"{value}\n" for value in values))


def _write_parquet(path: Path, values: list[int]) -> None:
    pq.write_table(pa.table({"x": values}), path)


def _write_parquet_with_file_path(
    path: Path, values: list[int], file_paths: list[str]
) -> None:
    pq.write_table(pa.table({"x": values, "file_path": file_paths}), path)


def _write_parquet_bytes(values: list[int]) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.table({"x": values}), buf)
    return buf.getvalue()


def _pipeline_values(pipeline: RefinerPipeline) -> list[int]:
    return [int(row["x"]) for row in pipeline.take(10)]


def test_datafileset_resolve_accepts_mixed_filesystems() -> None:
    with TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "local.jsonl"
        _write_jsonl(local_path, [1])

        memfs = MemoryFileSystem()
        memfs.pipe("remote.jsonl", b'{"x": 2}\n')

        fileset = DataFileSet.resolve(
            [str(local_path), DataFile(fs=memfs, path="remote.jsonl")]
        )

        assert len(fileset.entries) == 2
        assert len(fileset.files) == 2
        assert fileset.files[0].path == str(local_path)
        assert fileset.files[1].path == "remote.jsonl"


def test_datafileset_resolve_accepts_path_fs_tuple() -> None:
    memfs = MemoryFileSystem()
    memfs.pipe("remote.jsonl", b'{"x": 1}\n')

    fileset = DataFileSet.resolve([("remote.jsonl", memfs)])

    assert len(fileset.entries) == 1
    assert len(fileset.files) == 1
    assert fileset.files[0].fs is memfs


def test_datafileset_extensions_do_not_filter_explicit_files() -> None:
    with TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "local.txt"
        local_path.write_text("x")

        fileset = DataFileSet.resolve(str(local_path), extensions=(".jsonl",))

        assert len(fileset.files) == 1
        assert fileset.files[0].path == str(local_path)


def test_jsonl_reader_reads_across_multiple_directories() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        first_dir = root / "first"
        second_dir = root / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        _write_jsonl(first_dir / "a.jsonl", [1, 2])
        _write_jsonl(second_dir / "b.jsonl", [3])

        pipeline = read_json(
            [DataFolder(str(first_dir)), DataFolder(str(second_dir))], lines=True
        )

        assert _pipeline_values(pipeline) == [1, 2, 3]
        assert len(pipeline.source.list_shards()) == 1


def test_csv_reader_reads_across_mixed_local_and_memory_files() -> None:
    with TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "local.csv"
        _write_csv(local_path, [1])

        memfs = MemoryFileSystem()
        memfs.pipe("remote.csv", b"x\n2\n")

        pipeline = read_csv([str(local_path), DataFile(fs=memfs, path="remote.csv")])

        assert _pipeline_values(pipeline) == [1, 2]
        assert len(pipeline.source.list_shards()) == 1


def test_csv_reader_adds_file_path_column_by_default() -> None:
    with TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "local.csv"
        _write_csv(local_path, [1])

        row = read_csv(str(local_path)).take(1)[0].to_dict()

        assert row["x"] == 1
        assert row["file_path"] == str(local_path)


def test_parquet_reader_reads_across_mixed_local_and_memory_files() -> None:
    with TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "local.parquet"
        _write_parquet(local_path, [1])

        memfs = MemoryFileSystem()
        memfs.pipe("remote.parquet", _write_parquet_bytes([2]))

        pipeline = read_parquet(
            [str(local_path), DataFile(fs=memfs, path="remote.parquet")]
        )

        assert _pipeline_values(pipeline) == [1, 2]
        assert len(pipeline.source.list_shards()) == 1


def test_parquet_reader_filters_across_mixed_local_and_memory_files() -> None:
    with TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "local.parquet"
        _write_parquet(local_path, [1, 2])

        memfs = MemoryFileSystem()
        memfs.pipe("remote.parquet", _write_parquet_bytes([3, 4]))

        pipeline = read_parquet(
            [str(local_path), DataFile(fs=memfs, path="remote.parquet")],
            filter=col("x") > 2,
        )

        assert _pipeline_values(pipeline) == [3, 4]


def test_parquet_reader_can_disable_file_path_column() -> None:
    with TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "local.parquet"
        _write_parquet(local_path, [1])

        row = read_parquet(str(local_path), file_path_column=None).take(1)[0].to_dict()

        assert row == {"x": 1}


def test_parquet_reader_does_not_overwrite_existing_file_path_column() -> None:
    with TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "local.parquet"
        _write_parquet_with_file_path(local_path, [1], ["already-there"])

        row = read_parquet(str(local_path)).take(1)[0].to_dict()

        assert row["x"] == 1
        assert row["file_path"] == "already-there"


def test_parquet_reader_rejects_synthetic_file_path_in_projection() -> None:
    with TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "local.parquet"
        _write_parquet(local_path, [1])

        with pytest.raises(ValueError, match="synthetic file_path_column"):
            read_parquet(str(local_path), columns_to_read=["x", "file_path"])


@pytest.mark.parametrize("split_sources", [False, True])
def test_parquet_num_shards_is_exact(split_sources: bool) -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        first_dir = root / "first"
        second_dir = root / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        first_path = first_dir / "a.parquet"
        second_path = second_dir / "b.parquet"
        _write_parquet(first_path, list(range(10)))
        _write_parquet(second_path, list(range(10, 20)))
        inputs = (
            [str(first_path), str(second_path)]
            if not split_sources
            else [DataFolder(str(first_dir)), DataFolder(str(second_dir))]
        )
        shards = read_parquet(inputs, num_shards=5).source.list_shards()

        assert len(shards) == 5
