from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from typing import Any, TypeAlias, Union, cast

from fsspec import AbstractFileSystem, url_to_fs
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
