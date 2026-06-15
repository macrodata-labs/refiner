from __future__ import annotations

from collections.abc import Mapping
import os
from os import PathLike
import posixpath
import shutil
from typing import Any, TypeAlias, Union, cast

from fsspec import AbstractFileSystem, url_to_fs
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from refiner.io.utils import required_refiner_extras

DataFilePath: TypeAlias = str | PathLike[str]
DataFileSpec: TypeAlias = tuple[DataFilePath, AbstractFileSystem]
DataFileLike: TypeAlias = Union[DataFilePath, DataFileSpec, "DataFile"]
_HF_HTTP_PREFIXES = ("https://huggingface.co/", "https://hf.co/")


def _storage_options_for_path(
    path: str,
    storage_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    options = dict(storage_options or {})
    if path.startswith(_HF_HTTP_PREFIXES) and (hf_token := os.environ.get("HF_TOKEN")):
        headers = dict(cast(Mapping[str, str], options.get("headers") or {}))
        if not any(key.lower() == "authorization" for key in headers):
            headers["Authorization"] = f"Bearer {hf_token}"
            options["headers"] = headers
    return options


def _file_cache_key(file: "DataFile") -> tuple[object, str]:
    fs = file.fs
    fs_token = getattr(fs, "_fs_token", None)
    if fs_token is None:
        tokenize = getattr(fs, "__dask_tokenize__", None)
        fs_token = tokenize() if callable(tokenize) else id(fs)
    return (type(fs), fs_token), file.abs_path()


class DataFile:
    """A minimal (fs, path) file abstraction with a small normalization factory.

    Notes:
        - `path` is stored in the form expected by `fs.open/fs.exists` (no protocol required).
        - `resolve()` only normalizes `(fs, path)`; it does not check existence or list anything.
        - `resolve()` accepts `str` URL/path, `(path, fs)`, or `DataFile` (pass-through).
        - If `fs` is provided to `resolve()`, it wins and `storage_options` is ignored.
    """

    __slots__ = ("_fs", "_path", "_storage_options", "_abs_path")

    def __init__(
        self,
        fs: AbstractFileSystem | None,
        path: str,
        storage_options: Mapping[str, Any] | None = None,
    ) -> None:
        self._fs = fs
        # Keep string paths unresolved so cloud submission can inspect manifests
        # and infer extras without requiring local remote-storage credentials.
        self._path = (
            fs._strip_protocol(path) if fs is not None and "://" in path else path
        )
        self._storage_options = dict(storage_options or {})
        if fs is None:
            if "://" not in path and "::" not in path:
                self._abs_path = os.path.abspath(path)
            else:
                self._abs_path = path.removeprefix("file://")
        else:
            self._abs_path = fs.unstrip_protocol(self._path).removeprefix("file://")

    def _resolve(self) -> tuple[AbstractFileSystem, str]:
        if self._fs is None:
            self._fs, self._path = url_to_fs(
                self._path,
                **_storage_options_for_path(self._path, self._storage_options),
            )
        return self._fs, self._path

    @property
    def fs(self) -> AbstractFileSystem:
        return self._resolve()[0]

    @property
    def path(self) -> str:
        return self._resolve()[1]

    @classmethod
    def resolve(
        cls,
        data: DataFileLike,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
    ) -> "DataFile":
        """Resolve a path into a `DataFile`, or pass through an existing `DataFile`.

        Args:
            data: A URL/path, `(path, fs)` pair, or an existing `DataFile`.
            fs: Optional initialized filesystem to use. If provided, `storage_options` is ignored.
            storage_options: Optional fsspec filesystem init options (used only when `fs` is not provided).
        """
        if isinstance(data, cls):
            return data

        if (
            isinstance(data, tuple)
            and len(data) == 2
            and isinstance(data[1], AbstractFileSystem)
        ):
            spec = cast(DataFileSpec, data)
            data = spec[0]
            fs = spec[1]

        if isinstance(data, PathLike):
            data = str(data)

        # simple string url/path
        if isinstance(data, str):
            if fs is not None:
                return cls(fs=fs, path=data)

            return cls(fs=None, path=data, storage_options=storage_options)

        raise TypeError("DataFileLike must be: str | PathLike | (path, fs) | DataFile")

    def open(self, mode: str = "rt", **kwargs):
        return self.fs.open(self.path, mode=mode, **kwargs)

    def copy(self, dest: DataFileLike, *, buffer_size: int = 2 * 1024 * 1024) -> None:
        target = DataFile.resolve(dest)
        if self.abs_path() == target.abs_path():
            return
        if (
            self.is_local
            and target.is_local
            and os.path.realpath(self.abs_path()) == os.path.realpath(target.abs_path())
        ):
            return
        if self.fs is target.fs and (
            posixpath.normpath(self.path) == posixpath.normpath(target.path)
        ):
            return

        if self.is_local and callable(getattr(target.fs, "put_file", None)):
            target.fs.makedirs(target.fs._parent(target.path), exist_ok=True)
            try:
                target.fs.put_file(self.abs_path(), target.path)
                return
            except Exception:
                try:
                    target.fs.rm(target.path)
                except FileNotFoundError:
                    pass

        # Same-filesystem copies are usually server-side for object stores; fall back to
        # streaming only when the backend cannot copy directly.
        if self.fs is target.fs and callable(getattr(target.fs, "copy", None)):
            target.fs.makedirs(target.fs._parent(target.path), exist_ok=True)
            try:
                target.fs.copy(self.path, target.path)
                return
            except Exception:
                try:
                    target.fs.rm(target.path)
                except FileNotFoundError:
                    pass

        open_kwargs: dict[str, Any] = {}
        if isinstance(self.fs, HTTPFileSystem):
            open_kwargs = {"block_size": 0, "size": -1}
        elif self.fs.protocol == "hf":
            open_kwargs = {"block_size": 0}
        try:
            target.fs.makedirs(target.fs._parent(target.path), exist_ok=True)
            with (
                self.open(mode="rb", **open_kwargs) as src,
                target.open(mode="wb") as dst,
            ):
                shutil.copyfileobj(src, dst, length=buffer_size)
        except Exception:
            try:
                target.fs.rm(target.path)
            except FileNotFoundError:
                pass
            raise

    def exists(self) -> bool:
        return self.fs.exists(self.path)

    def abs_path(self) -> str:
        return self._abs_path

    def required_refiner_extras(self) -> tuple[str, ...]:
        return required_refiner_extras(self._path, self._fs)

    @property
    def is_local(self) -> bool:
        return isinstance(self.fs, LocalFileSystem)

    def __str__(self) -> str:
        return self.abs_path()
