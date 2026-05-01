from pathlib import Path
from typing import Any

import pytest

from refiner import datatype, read_files
from refiner.io import DataFile
import refiner.pipeline.sources.readers.files as files_reader_module
from refiner.pipeline.sources.readers.files import FilesReader


def _row_dicts(path: str, **kwargs) -> list[dict]:
    return [row.to_dict() for row in read_files(path, **kwargs).materialize()]


def test_read_files_lists_paths_without_opening_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    def fail_open(*args, **kwargs):
        raise AssertionError("path-only read_files should not open files")

    monkeypatch.setattr(DataFile, "open", fail_open)

    rows = _row_dicts(str(tmp_path))

    assert rows == [
        {"file_path": str(first)},
        {"file_path": str(second)},
    ]


def test_read_files_reads_binary_content_from_glob(tmp_path: Path) -> None:
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    third = tmp_path / "c.bin"
    ignored = tmp_path / "ignored.txt"
    first.write_bytes(b"\x00first")
    second.write_bytes(b"\xffsecond")
    third.write_bytes(b"third")
    ignored.write_bytes(b"ignored")

    rows = _row_dicts(
        str(tmp_path / "*.bin"),
        content_column="content",
        max_in_flight=2,
    )

    assert rows == [
        {"file_path": str(first), "content": b"\x00first"},
        {"file_path": str(second), "content": b"\xffsecond"},
        {"file_path": str(third), "content": b"third"},
    ]


def test_read_files_supports_recursive_directory_listing(tmp_path: Path) -> None:
    top = tmp_path / "top.txt"
    nested = tmp_path / "nested" / "leaf.txt"
    nested.parent.mkdir()
    top.write_bytes(b"top")
    nested.write_bytes(b"leaf")

    rows = _row_dicts(str(tmp_path), recursive=True)

    assert rows == [
        {"file_path": str(nested)},
        {"file_path": str(top)},
    ]


def test_read_files_can_emit_content_without_path_column(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")

    rows = _row_dicts(
        str(path),
        file_path_column=None,
        content_column="body",
        max_in_flight=1,
    )

    assert rows == [{"body": b"payload"}]


def test_read_files_rejects_empty_output_rows(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")

    with pytest.raises(ValueError, match="file_path_column or content_column"):
        read_files(str(path), file_path_column=None)


def test_read_files_cancels_pending_chunk_when_content_read_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    windows: list[Any] = []

    class RecordingWindow(files_reader_module.AsyncWindow):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.cancelled = False
            windows.append(self)

        def cancel_pending(self) -> None:
            self.cancelled = True
            super().cancel_pending()

    monkeypatch.setattr(files_reader_module, "AsyncWindow", RecordingWindow)

    reader = read_files(
        [str(first), str(second)],
        content_column="content",
        max_in_flight=2,
    ).source
    assert isinstance(reader, FilesReader)
    original = reader._content_row

    def fail_on_second(source: DataFile) -> dict[str, Any]:
        if source.path == str(second):
            raise RuntimeError("boom")
        return original(source)

    monkeypatch.setattr(reader, "_content_row", fail_on_second)

    with pytest.raises(RuntimeError, match="boom"):
        list(reader.read_shard(reader.list_shards()[0]))

    assert windows
    assert windows[0].cancelled


def test_read_files_schema_exposes_dtype_override_for_content_column(
    tmp_path: Path,
) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")

    schema = read_files(
        str(path),
        content_column="body",
        dtypes={"body": datatype.file_bytes()},
    ).source.schema

    assert schema is not None
    assert schema.field("body").metadata == {b"asset_type": b"file"}
