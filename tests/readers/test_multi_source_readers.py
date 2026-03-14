from __future__ import annotations

import io
from pathlib import Path
from tempfile import TemporaryDirectory

from fsspec.implementations.memory import MemoryFileSystem
import pyarrow as pa
import pyarrow.parquet as pq

from refiner.io import DataFile, DataFolder
from refiner.io.fileset import DataFileSet
from refiner.pipeline import RefinerPipeline, read_csv, read_jsonl, read_parquet


def _write_jsonl(path: Path, values: list[int]) -> None:
    path.write_text("".join(f'{{"x": {value}}}\n' for value in values))


def _write_csv(path: Path, values: list[int]) -> None:
    path.write_text("x\n" + "".join(f"{value}\n" for value in values))


def _write_parquet(path: Path, values: list[int]) -> None:
    pq.write_table(pa.table({"x": values}), path)


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

        assert len(fileset.files) == 2
        assert fileset.files[0].path == str(local_path)
        assert fileset.files[1].path == "remote.jsonl"


def test_jsonl_reader_reads_across_multiple_directories() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        first_dir = root / "first"
        second_dir = root / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        _write_jsonl(first_dir / "a.jsonl", [1, 2])
        _write_jsonl(second_dir / "b.jsonl", [3])

        pipeline = read_jsonl([DataFolder(str(first_dir)), DataFolder(str(second_dir))])

        assert _pipeline_values(pipeline) == [1, 2, 3]
        assert len(pipeline.source.list_shards()) == 2


def test_csv_reader_reads_across_mixed_local_and_memory_files() -> None:
    with TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "local.csv"
        _write_csv(local_path, [1])

        memfs = MemoryFileSystem()
        memfs.pipe("remote.csv", b"x\n2\n")

        pipeline = read_csv([str(local_path), DataFile(fs=memfs, path="remote.csv")])

        assert _pipeline_values(pipeline) == [1, 2]
        assert len(pipeline.source.list_shards()) == 2


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
        assert len(pipeline.source.list_shards()) == 2
