from __future__ import annotations

import io
from pathlib import Path
import tarfile
from typing import Literal

from fsspec.implementations.memory import MemoryFileSystem
import pytest

from refiner.io import DataFile
from refiner.pipeline import read_webdataset
from refiner.pipeline.data import datatype
from refiner.pipeline.sources.readers.webdataset import WebDatasetReader


def _tar_bytes(
    members: list[tuple[str, bytes]],
    *,
    mode: Literal["w", "w:gz"] = "w",
) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode=mode) as tar:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return out.getvalue()


def _write_tar(
    path: Path,
    members: list[tuple[str, bytes]],
    *,
    mode: Literal["w", "w:gz"] = "w",
) -> None:
    path.write_bytes(_tar_bytes(members, mode=mode))


def test_webdataset_reader_groups_members_into_samples(tmp_path: Path) -> None:
    path = tmp_path / "samples.tar"
    _write_tar(
        path,
        [
            ("0001.jpg", b"image-1"),
            ("0001.json", b'{"label": "cat", "score": 3}'),
            ("0002.jpg", b"image-2"),
            ("0002.txt", b"caption"),
        ],
    )

    rows = read_webdataset(str(path)).take(10)

    assert [row["sample_key"] for row in rows] == ["0001", "0002"]
    assert rows[0]["file_path"] == str(path)
    assert rows[0]["jpg"] == b"image-1"
    assert rows[0]["json"] == {"label": "cat", "score": 3}
    assert rows[1]["jpg"] == b"image-2"
    assert rows[1]["txt"] == b"caption"


def test_webdataset_reader_normalizes_dot_prefixed_member_paths(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dot-prefix.tar"
    _write_tar(
        path,
        [
            ("./0001.jpg", b"image"),
            ("./0001.json", b'{"label": "cat"}'),
        ],
    )

    row = read_webdataset(str(path), file_path_column=None).take(1)[0]

    assert row["sample_key"] == "0001"
    assert row["jpg"] == b"image"
    assert row["json"] == {"label": "cat"}


def test_webdataset_reader_uses_suffix_after_first_dot_as_field_name(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compound-fields.tar"
    _write_tar(
        path,
        [
            ("0001.jpg", b"image"),
            ("0001.seg.png", b"mask"),
        ],
    )

    row = read_webdataset(str(path), file_path_column=None).take(1)[0]

    assert row["sample_key"] == "0001"
    assert row["jpg"] == b"image"
    assert row["seg.png"] == b"mask"


def test_webdataset_reader_preserves_nested_sample_key_prefixes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested.tar"
    _write_tar(
        path,
        [
            ("split/a.0001.png", b"png"),
            ("split/a.0001.json", b'{"id": 1}'),
        ],
    )

    row = read_webdataset(str(path), file_path_column=None).take(1)[0]

    assert row["sample_key"] == "split/a"
    assert row["0001.png"] == b"png"
    assert row["0001.json"] == {"id": 1}


def test_webdataset_reader_can_return_json_bytes(tmp_path: Path) -> None:
    path = tmp_path / "bytes.tar"
    _write_tar(path, [("0001.json", b'{"raw": true}')])

    row = read_webdataset(str(path), parse_json=False).take(1)[0]

    assert row["json"] == b'{"raw": true}'


def test_webdataset_reader_reads_gzipped_archives_from_folders(tmp_path: Path) -> None:
    _write_tar(tmp_path / "a.tar.gz", [("0001.txt", b"a")], mode="w:gz")
    _write_tar(tmp_path / "b.tgz", [("0002.txt", b"b")], mode="w:gz")
    (tmp_path / "ignore.txt").write_text("not an archive")

    rows = read_webdataset(str(tmp_path), file_path_column=None).take(10)

    assert [(row["sample_key"], row["txt"]) for row in rows] == [
        ("0001", b"a"),
        ("0002", b"b"),
    ]


def test_webdataset_reader_accepts_fsspec_datafiles() -> None:
    memfs = MemoryFileSystem()
    memfs.pipe("remote.tar", _tar_bytes([("0001.bin", b"payload")]))

    row = read_webdataset(DataFile(fs=memfs, path="remote.tar")).take(1)[0]

    assert row["sample_key"] == "0001"
    assert row["bin"] == b"payload"
    assert row["file_path"] == "memory://remote.tar"


def test_webdataset_reader_keeps_archives_atomic(tmp_path: Path) -> None:
    path = tmp_path / "samples.tar"
    _write_tar(path, [("0001.txt", b"x")])

    shards = read_webdataset(str(path), num_shards=4).source.list_shards()

    assert len(shards) == 1


def test_webdataset_reader_exposes_dtype_overrides() -> None:
    reader = WebDatasetReader(
        "missing.tar",
        dtypes={"jpg": datatype.image_bytes(), "sample_key": datatype.string()},
    )

    assert reader.schema is not None
    assert reader.schema.field("jpg").metadata == {b"asset_type": b"image"}
    assert reader.describe()["dtypes"] == {
        "jpg": {"type": "binary", "metadata": {"asset_type": "image"}},
        "sample_key": "string",
    }


def test_webdataset_reader_rejects_duplicate_metadata_columns() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        WebDatasetReader(
            "missing.tar", file_path_column="path", sample_key_column="path"
        )


def test_webdataset_reader_rejects_member_metadata_collision(tmp_path: Path) -> None:
    path = tmp_path / "collision.tar"
    _write_tar(path, [("0001.sample_key", b"x")])

    with pytest.raises(ValueError, match="collides with a metadata column"):
        read_webdataset(str(path)).take(1)
