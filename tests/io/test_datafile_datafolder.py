from fsspec.implementations.memory import MemoryFileSystem
from fsspec.implementations.local import LocalFileSystem

from refiner.io.datafile import DataFile
from refiner.io.datafolder import DataFolder
from refiner.io.fileset import DataFileSet


def test_datafile_resolve_with_path_string(tmp_path):
    p = tmp_path / "hello.txt"
    p.write_text("hi")

    df = DataFile.resolve(str(p))
    assert df.exists()
    assert df.is_local
    assert df.abs_path() == str(p)


def test_datafile_resolve_with_fs(tmp_path):
    fs = LocalFileSystem()
    p = tmp_path / "hello.txt"
    p.write_text("hi")

    df = DataFile.resolve(str(p), fs=fs)
    assert df.exists()
    assert df.is_local


def test_datafolder_file_and_open_creates_parent_dirs(tmp_path):
    folder = DataFolder.resolve(str(tmp_path))

    # Opening nested path in write mode should auto-mkdir.
    with folder.open("nested/dir/out.txt", mode="wt") as f:
        f.write("ok")

    assert folder.exists("nested/dir/out.txt")

    df = folder.file("nested/dir/out.txt")
    assert df.exists()


class _CountingMemoryFS(MemoryFileSystem):
    def __init__(self):
        super().__init__()
        self.ls_calls = 0

    def ls(self, path, detail=True, **kwargs):
        self.ls_calls += 1
        return super().ls(path, detail=detail, **kwargs)


def test_datafileset_resolve_does_not_list_folders_eagerly():
    fs = _CountingMemoryFS()
    fs.pipe("root/a.txt", b"a")

    fileset = DataFileSet.resolve(DataFolder("root", fs=fs))

    assert fs.ls_calls == 0
    assert len(fileset.files) == 1
    assert fileset.files[0].open().read() == "a"
    assert fs.ls_calls == 1
