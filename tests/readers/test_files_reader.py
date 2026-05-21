from pathlib import Path
from typing import Any, cast

import pytest

from refiner import datatype, read_files, read_videos
from refiner.io import DataFile
from refiner.pipeline.data.row import Row
from refiner.pipeline import read_videos as pipeline_read_videos
import refiner.pipeline.sources.readers.files as files_reader_module
import refiner.pipeline.pipeline as pipeline_module
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
        {"file_path": str(first), "size": 5},
        {"file_path": str(second), "size": 6},
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
        {"file_path": str(first), "size": 6, "content": b"\x00first"},
        {"file_path": str(second), "size": 7, "content": b"\xffsecond"},
        {"file_path": str(third), "size": 5, "content": b"third"},
    ]


def test_read_files_supports_recursive_directory_listing(tmp_path: Path) -> None:
    top = tmp_path / "top.txt"
    nested = tmp_path / "nested" / "leaf.txt"
    nested.parent.mkdir()
    top.write_bytes(b"top")
    nested.write_bytes(b"leaf")

    rows = _row_dicts(str(tmp_path), recursive=True)

    assert rows == [
        {"file_path": str(nested), "size": 4},
        {"file_path": str(top), "size": 3},
    ]


def test_read_files_ignores_max_in_flight_for_path_only_rows(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")

    rows = _row_dicts(str(path), max_in_flight=0)

    assert rows == [{"file_path": str(path), "size": 7}]


def test_read_files_can_emit_content_without_path_column(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")

    rows = _row_dicts(
        str(path),
        file_path_column=None,
        content_column="body",
        size_column=None,
        max_in_flight=1,
    )

    assert rows == [{"body": b"payload"}]


def test_read_files_validates_max_in_flight_when_reading_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")

    with pytest.raises(ValueError, match="max_in_flight"):
        read_files(str(path), content_column="body", max_in_flight=0)


def test_read_files_can_decode_content(tmp_path: Path) -> None:
    path = tmp_path / "payload.txt"
    path.write_bytes(b"payload")

    rows = _row_dicts(
        str(path),
        content_column="text",
        decode_fn=lambda data: data.decode("utf-8"),
    )

    assert rows == [{"file_path": str(path), "size": 7, "text": "payload"}]


def test_read_files_rejects_empty_output_rows(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")

    with pytest.raises(ValueError, match="requires file_path_column"):
        read_files(str(path), file_path_column=None, size_column=None)


def test_read_files_rejects_decode_without_content_column(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")

    with pytest.raises(ValueError, match="decode_fn requires content_column"):
        read_files(str(path), decode_fn=lambda data: data)


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

    def fail_on_second(source: DataFile, part) -> dict[str, Any]:
        if source.path == str(second):
            raise RuntimeError("boom")
        return original(source, part)

    monkeypatch.setattr(reader, "_content_row", fail_on_second)

    with pytest.raises(RuntimeError, match="boom"):
        list(reader.read_shard(reader.list_shards()[0]))

    assert windows
    assert windows[0].cancelled


def test_read_files_cancels_pending_reads_when_generator_closes(
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

    iterator = cast(Any, reader.read_shard(reader.list_shards()[0]))
    next(iterator)
    iterator.close()

    assert windows
    assert windows[0].cancelled


def test_read_files_preserves_planned_size_in_shard_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")
    reader = read_files(str(path)).source
    assert isinstance(reader, FilesReader)
    shard = reader.list_shards()[0]
    roundtripped = type(shard).from_dict(shard.to_dict())

    def fail_size(*args, **kwargs):
        raise AssertionError("read_files should reuse planned shard sizes")

    monkeypatch.setattr(type(reader.fileset), "size", fail_size)

    rows: list[dict[str, Any]] = []
    for row in reader.read_shard(roundtripped):
        assert isinstance(row, Row)
        rows.append(row.to_dict())

    assert rows == [{"file_path": str(path), "size": 7}]


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


def test_read_videos_defaults_to_video_path_column(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")

    pipeline = read_videos(str(path))
    schema = pipeline.source.schema
    rows = [row.to_dict() for row in pipeline.materialize()]

    assert rows == [{"video_path": str(path), "size": 5}]
    assert schema is not None
    assert schema.field("video_path").metadata == {b"asset_type": b"video"}
    assert read_videos is pipeline_read_videos


def test_read_videos_marks_custom_path_column_as_video(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")

    schema = read_videos(
        str(path),
        file_path_column="source_video",
        dtypes={"label": datatype.large_string()},
    ).source.schema

    assert schema is not None
    assert schema.field("source_video").metadata == {b"asset_type": b"video"}
    assert schema.field("label").type == datatype.large_string()


def test_read_videos_preserves_explicit_path_column_dtype(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")

    schema = read_videos(
        str(path),
        dtypes={"video_path": datatype.file_path()},
    ).source.schema

    assert schema is not None
    assert schema.field("video_path").metadata == {b"asset_type": b"file"}


def test_read_videos_forwards_file_listing_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_read_files(inputs, **kwargs):
        captured["inputs"] = inputs
        captured.update(kwargs)
        return "pipeline"

    monkeypatch.setattr(pipeline_module, "read_files", fake_read_files)

    pipeline = read_videos(
        "videos/**/*.mp4",
        fs=cast(Any, "fs"),
        storage_options={"token": "secret"},
        recursive=True,
        target_shard_bytes=123,
        num_shards=4,
        file_path_column="clip",
        size_column=None,
        dtypes={"label": datatype.large_string()},
    )

    assert pipeline == "pipeline"
    assert captured == {
        "inputs": "videos/**/*.mp4",
        "fs": "fs",
        "storage_options": {"token": "secret"},
        "recursive": True,
        "target_shard_bytes": 123,
        "num_shards": 4,
        "file_path_column": "clip",
        "size_column": None,
        "dtypes": {
            "label": datatype.large_string(),
            "clip": datatype.video_path(),
        },
    }
