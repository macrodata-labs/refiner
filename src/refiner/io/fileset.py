from __future__ import annotations

import glob
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, TypeAlias, Union

from fsspec import AbstractFileSystem, url_to_fs

from refiner.io.datafile import DataFile
from refiner.io.datafolder import DataFolder

DataFileSetInput: TypeAlias = Union[
    str, PathLike[str], DataFile, DataFolder, "DataFileSet"
]
DataFileSetLike: TypeAlias = Union[DataFileSetInput, Sequence[DataFileSetInput]]


@dataclass(frozen=True, slots=True)
class DataFileSet:
    """A deterministic set of resolved input files.

    Notes:
        - Each entry is a resolved `DataFile` carrying its own filesystem and fs-native path.
        - Input order is preserved across heterogeneous sources; files discovered within one
          input are sorted deterministically before being appended.
    """

    files: tuple[DataFile, ...]
    _sizes: dict[int, int] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @classmethod
    def resolve(
        cls,
        data: DataFileSetLike,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        recursive: bool = False,
        extensions: Sequence[str] | None = None,
    ) -> "DataFileSet":
        if isinstance(data, cls):
            return data

        if isinstance(data, (str, PathLike, DataFile, DataFolder, DataFileSet)):
            inputs = (data,)
        else:
            inputs = tuple(data)

        storage_options_d = dict(storage_options or {})
        exts = tuple(e.lower() for e in (extensions or ()))

        resolved_files: list[DataFile] = []
        seen: set[tuple[int, str]] = set()

        def _dedupe_key(file: DataFile) -> tuple[int, str]:
            return id(file.fs), file.path

        def _append_files(next_fs: AbstractFileSystem, paths: Iterable[str]) -> None:
            for p in paths:
                if exts and not p.lower().endswith(exts):
                    continue
                if next_fs.exists(p) and not next_fs.isdir(p):
                    file = DataFile(fs=next_fs, path=p)
                    key = _dedupe_key(file)
                    if key in seen:
                        continue
                    seen.add(key)
                    resolved_files.append(file)

        def _list_dir(next_fs: AbstractFileSystem, dir_path: str) -> Iterable[str]:
            if recursive:
                return sorted(next_fs.find(dir_path))

            entries = next_fs.ls(dir_path, detail=True)
            files: list[str] = []
            for e in entries:
                if isinstance(e, dict):
                    name = e.get("name")
                    typ = e.get("type")
                    if isinstance(name, str) and typ == "file":
                        files.append(name)
                elif isinstance(e, str):
                    files.append(e)
            return sorted(files)

        for item in inputs:
            if isinstance(item, DataFileSet):
                for file in item.files:
                    key = _dedupe_key(file)
                    if key in seen:
                        continue
                    seen.add(key)
                    resolved_files.append(file)
                continue

            if isinstance(item, DataFile):
                _append_files(item.fs, [item.path])
                continue

            if isinstance(item, DataFolder):
                _append_files(item.fs, _list_dir(item.fs, item.path))
                continue

            if isinstance(item, PathLike):
                item = str(item)

            if not isinstance(item, str):
                raise TypeError(
                    "DataFileSet inputs must be str | PathLike | DataFile | DataFolder | DataFileSet"
                )

            if fs is not None:
                next_fs = fs
                p = next_fs._strip_protocol(item)
            else:
                next_fs, p = url_to_fs(item, **storage_options_d)

            if glob.has_magic(p):
                _append_files(next_fs, sorted(next_fs.glob(p)))
                continue

            if next_fs.exists(p):
                if next_fs.isdir(p):
                    _append_files(next_fs, _list_dir(next_fs, p))
                else:
                    _append_files(next_fs, [p])
                continue

            raise FileNotFoundError(f"Could not resolve input: {item!r}")

        out = cls(files=tuple(resolved_files))
        for index, file in enumerate(out.files):
            try:
                out._sizes[index] = int(file.fs.size(file.path))
            except Exception:
                continue
        return out

    def size(self, index: int) -> int:
        """Return the size for a resolved file entry, caching results."""
        target = self.files[index]
        if index in self._sizes:
            return self._sizes[index]
        sz = int(target.fs.size(target.path))
        self._sizes[index] = sz
        return sz


__all__ = ["DataFileSet", "DataFileSetLike"]
