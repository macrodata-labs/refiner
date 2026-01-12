from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from fsspec import AbstractFileSystem, url_to_fs
from fsspec.implementations.local import LocalFileSystem

DataFileLike: TypeAlias = (
    str
    | tuple[str, Mapping[str, Any]]
    | tuple[str, AbstractFileSystem]  # (path, fs)
    | "DataFile"
)


@dataclass(frozen=True, slots=True)
class DataFile:
    """A minimal (fs, path) file abstraction with a small normalization factory.

    Notes:
        - `path` is stored in the form expected by `fs.open/fs.exists` (no protocol required).
        - `resolve()` accepts:
            * `str` URL/path
            * `(str, Mapping)` interpreted as `(url, storage_options)`
            * `(str, AbstractFileSystem)` interpreted as `(path, fs)` (your existing convention)
            * `DataFile` (pass-through)
    """

    fs: AbstractFileSystem
    path: str

    @classmethod
    def resolve(cls, data: DataFileLike) -> "DataFile":
        if isinstance(data, cls):
            return data

        # simple string url/path
        if isinstance(data, str):
            fs, path = url_to_fs(data)
            return cls(fs=fs, path=path)

        # (url/path, storage_options)
        if (
            isinstance(data, tuple)
            and len(data) == 2
            and isinstance(data[0], str)
            and isinstance(data[1], Mapping)
        ):
            fs, path = url_to_fs(data[0], **cast(Mapping[str, Any], data[1]))
            return cls(fs=fs, path=path)

        # (path, initialized fs)
        if (
            isinstance(data, tuple)
            and len(data) == 2
            and isinstance(data[0], str)
            and isinstance(data[1], AbstractFileSystem)
        ):
            return cls(fs=data[1], path=data[0])

        raise TypeError(
            "DataFileLike must be: str | (str, Mapping) | (str, AbstractFileSystem) | DataFile"
        )

    def open(self, mode: str = "rt", **kwargs):
        return self.fs.open(self.path, mode=mode, **kwargs)

    def exists(self) -> bool:
        return self.fs.exists(self.path)

    @property
    def is_local(self) -> bool:
        return isinstance(self.fs, LocalFileSystem)

    def __str__(self) -> str:
        try:
            return self.fs.unstrip_protocol(self.path)
        except Exception:
            return self.path
