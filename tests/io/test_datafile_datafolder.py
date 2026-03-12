from fsspec.implementations.local import LocalFileSystem

from refiner.io.datafile import DataFile
from refiner.io.datafolder import DataFolder


def test_datafile_resolve_with_path_string(tmp_path):
    p = tmp_path / "hello.txt"
    p.write_text("hi")

    df = DataFile.resolve(str(p))
    assert df.exists()
    assert df.is_local


def test_datafile_resolve_with_fs(tmp_path):
    fs = LocalFileSystem()
    p = tmp_path / "hello.txt"
    p.write_text("hi")

    df = DataFile.resolve(str(p), fs=fs)
    assert df.exists()
    assert df.is_local


def test_datafile_cat_reads_bytes(tmp_path):
    p = tmp_path / "hello.bin"
    p.write_bytes(b"abc123")

    df = DataFile.resolve(str(p))

    assert df.cat() == b"abc123"


def test_datafolder_file_and_open_creates_parent_dirs(tmp_path):
    folder = DataFolder.resolve(str(tmp_path))

    # Opening nested path in write mode should auto-mkdir.
    with folder.open("nested/dir/out.txt", mode="wt") as f:
        f.write("ok")

    assert folder.exists("nested/dir/out.txt")

    df = folder.file("nested/dir/out.txt")
    assert df.exists()
