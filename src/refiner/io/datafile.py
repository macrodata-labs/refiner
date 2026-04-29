from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from os import PathLike
import posixpath
import shutil
from typing import Any, TypeAlias, Union, cast

from fsspec import AbstractFileSystem, url_to_fs
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem

DataFilePath: TypeAlias = str | PathLike[str]
DataFileSpec: TypeAlias = tuple[DataFilePath, AbstractFileSystem]
DataFileLike: TypeAlias = Union[DataFilePath, DataFileSpec, "DataFile"]


@dataclass(frozen=True, slots=True)
class DataFile:
    """A minimal (fs, path) file abstraction with a small normalization factory.

    Notes:
        - `path` is stored in the form expected by `fs.open/fs.exists` (no protocol required).
        - `resolve()` only normalizes `(fs, path)`; it does not check existence or list anything.
        - `resolve()` accepts `str` URL/path, `(path, fs)`, or `DataFile` (pass-through).
        - If `fs` is provided to `resolve()`, it wins and `storage_options` is ignored.
    """

    fs: AbstractFileSystem
    path: str

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
                # Best-effort strip protocol so `.path` is in the form expected by `fs.open/fs.exists`.
                path = fs._strip_protocol(data)
                return cls(fs=fs, path=path)

            next_fs, path = url_to_fs(
                data, **cast(Mapping[str, Any], storage_options or {})
            )
            return cls(fs=next_fs, path=path)

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
        return self.fs.unstrip_protocol(self.path).removeprefix("file://")

    @property
    def is_local(self) -> bool:
        return isinstance(self.fs, LocalFileSystem)

    def __str__(self) -> str:
        try:
            return self.fs.unstrip_protocol(self.path)
        except Exception:
            return self.path
