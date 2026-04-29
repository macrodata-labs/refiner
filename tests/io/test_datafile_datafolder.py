from fsspec.implementations.memory import MemoryFileSystem
from fsspec.implementations.local import LocalFileSystem
from typing import Any, cast

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


def test_datafile_resolve_with_path_fs_tuple(tmp_path):
    fs = LocalFileSystem()
    p = tmp_path / "hello.txt"
    p.write_text("hi")

    df = DataFile.resolve((p, fs))
    assert df.exists()
    assert df.is_local


def test_datafile_copy_skips_same_source_and_destination(tmp_path):
    asset = tmp_path / "image.png"
    asset.write_bytes(b"existing")
    source = DataFile.resolve(str(asset))

    source.copy(str(asset))
    assert asset.read_bytes() == b"existing"

    source.copy(str(tmp_path / "nested" / ".." / "image.png"))
    assert asset.read_bytes() == b"existing"


def test_datafile_copy_writes_destination(tmp_path):
    source_path = tmp_path / "source.txt"
    dest_path = tmp_path / "nested" / "dest.txt"
    source_path.write_bytes(b"payload")

    DataFile.resolve(str(source_path)).copy(str(dest_path))

    assert dest_path.read_bytes() == b"payload"


def test_datafile_resolve_adds_hf_token_for_huggingface_http_urls(monkeypatch):
    captured = {}

    def fake_url_to_fs(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return MemoryFileSystem(), path

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr("refiner.io.datafile.url_to_fs", fake_url_to_fs)

    DataFile.resolve("https://huggingface.co/datasets/org/repo/resolve/main/file.mp4")

    assert captured["kwargs"]["headers"] == {"Authorization": "Bearer hf_test"}


def test_datafile_resolve_adds_hf_token_for_hf_short_urls(monkeypatch):
    captured = {}

    def fake_url_to_fs(path, **kwargs):
        captured["kwargs"] = kwargs
        return MemoryFileSystem(), path

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr("refiner.io.datafile.url_to_fs", fake_url_to_fs)

    DataFile.resolve("https://hf.co/datasets/org/repo/resolve/main/file.mp4")

    assert captured["kwargs"]["headers"] == {"Authorization": "Bearer hf_test"}


def test_datafile_resolve_preserves_explicit_headers_for_huggingface_http_urls(
    monkeypatch,
):
    captured = {}

    def fake_url_to_fs(path, **kwargs):
        captured["kwargs"] = kwargs
        return MemoryFileSystem(), path

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr("refiner.io.datafile.url_to_fs", fake_url_to_fs)

    DataFile.resolve(
        "https://huggingface.co/datasets/org/repo/resolve/main/file.mp4",
        storage_options={"headers": {"Authorization": "Bearer explicit"}},
    )

    assert captured["kwargs"]["headers"] == {"Authorization": "Bearer explicit"}


def test_datafile_resolve_merges_hf_token_with_existing_headers(monkeypatch):
    captured = {}

    def fake_url_to_fs(path, **kwargs):
        captured["kwargs"] = kwargs
        return MemoryFileSystem(), path

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr("refiner.io.datafile.url_to_fs", fake_url_to_fs)

    DataFile.resolve(
        "https://huggingface.co/datasets/org/repo/resolve/main/file.mp4",
        storage_options={"headers": {"User-Agent": "refiner-test"}},
    )

    assert captured["kwargs"]["headers"] == {
        "User-Agent": "refiner-test",
        "Authorization": "Bearer hf_test",
    }


def test_datafile_resolve_preserves_lowercase_authorization_header(monkeypatch):
    captured = {}

    def fake_url_to_fs(path, **kwargs):
        captured["kwargs"] = kwargs
        return MemoryFileSystem(), path

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr("refiner.io.datafile.url_to_fs", fake_url_to_fs)

    DataFile.resolve(
        "https://huggingface.co/datasets/org/repo/resolve/main/file.mp4",
        storage_options={"headers": {"authorization": "Bearer explicit"}},
    )

    assert captured["kwargs"]["headers"] == {"authorization": "Bearer explicit"}


def test_datafolder_resolve_adds_hf_token_for_huggingface_http_urls(monkeypatch):
    captured = {}

    def fake_url_to_fs(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return MemoryFileSystem(), path

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr("refiner.io.datafolder.url_to_fs", fake_url_to_fs)

    DataFolder.resolve("https://huggingface.co/datasets/org/repo/tree/main/assets")

    assert captured["kwargs"]["headers"] == {"Authorization": "Bearer hf_test"}


def test_datafileset_resolve_adds_hf_token_for_generic_hf_http_urls(monkeypatch):
    captured = {}

    def fake_url_to_fs(path, **kwargs):
        captured["kwargs"] = kwargs
        return MemoryFileSystem(), path

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr("refiner.io.fileset.url_to_fs", fake_url_to_fs)

    DataFileSet.resolve("https://huggingface.co/datasets/org/repo/resolve/main/file")

    assert captured["kwargs"]["headers"] == {"Authorization": "Bearer hf_test"}


def test_datafolder_file_and_open_creates_parent_dirs(tmp_path):
    folder = DataFolder.resolve(str(tmp_path))

    # Opening nested path in write mode should auto-mkdir.
    with folder.open("nested/dir/out.txt", mode="wt") as f:
        f.write("ok")

    assert folder.exists("nested/dir/out.txt")

    df = folder.file("nested/dir/out.txt")
    assert df.exists()


def test_datafolder_resolve_with_path_fs_tuple(tmp_path):
    fs = LocalFileSystem()
    folder = DataFolder.resolve((tmp_path, fs))

    with folder.open("out.txt", mode="wt") as f:
        f.write("ok")

    assert folder.exists("out.txt")


class _CountingMemoryFS(MemoryFileSystem):
    def __init__(self):
        super().__init__()
        self.ls_calls = 0

    def ls(self, path, detail=True, **kwargs):
        self.ls_calls += 1
        return super().ls(path, detail=detail, **kwargs)


class _PrefixLeakingMemoryFS(MemoryFileSystem):
    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        leaked = super().find(
            path,
            maxdepth=maxdepth,
            withdirs=withdirs,
            detail=detail,
            **kwargs,
        )
        sibling = f"{path.rstrip('/')}-2/other.txt"
        if detail:
            leaked[sibling] = {
                "name": sibling,
                "size": 1,
                "type": "file",
                "created": 0,
            }
            return leaked
        return [*leaked, sibling]


def test_datafileset_resolve_does_not_list_folders_eagerly():
    fs = _CountingMemoryFS()
    fs.pipe("root/a.txt", b"a")

    fileset = DataFileSet.resolve(DataFolder("root", fs=fs))

    assert fs.ls_calls == 0
    assert len(fileset.files) == 1
    assert fileset.files[0].open().read() == "a"
    assert fs.ls_calls == 1


def test_datafolder_find_filters_prefix_leaks():
    fs = _PrefixLeakingMemoryFS()
    fs.pipe("find-root/file.txt", b"ok")
    folder = DataFolder.resolve(("find-root", fs))

    assert folder.find("") == ["file.txt"]
    assert folder.find("", withdirs=True) == ["", "file.txt"]


def test_datafolder_find_preserves_file_paths():
    fs = MemoryFileSystem()
    fs.pipe("find-file-root/file.txt", b"ok")
    folder = DataFolder.resolve(("find-file-root", fs))

    assert folder.find("file.txt") == ["file.txt"]
    detailed = folder.find("file.txt", detail=True)
    assert list(detailed) == ["file.txt"]
    assert detailed["file.txt"]["type"] == "file"


def test_datafolder_find_accepts_backend_paths_without_leading_separator():
    fs = MemoryFileSystem()
    fs.pipe("find-slash-root/file.txt", b"ok")
    folder = DataFolder("/find-slash-root", fs=fs)

    assert folder.find("") == ["file.txt"]
    assert folder.find("file.txt") == ["file.txt"]


def test_datafileset_rejects_nested_filesets():
    nested = DataFileSet.resolve(["a.txt"])

    try:
        DataFileSet.resolve(cast(Any, [nested]))
    except TypeError as exc:
        assert "DataFile | DataFolder" in str(exc)
        return

    raise AssertionError("Expected nested DataFileSet inputs to be rejected")
