from __future__ import annotations

import glob
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, TypeAlias, Union

from fsspec import AbstractFileSystem, url_to_fs
from fsspec.implementations.local import LocalFileSystem

from refiner.io.datafile import DataFile
from refiner.io.datafolder import DataFolder

DataFileSetInput: TypeAlias = Union[
    str, PathLike[str], DataFile, DataFolder, "DataFileSet"
]
DataFileSetLike: TypeAlias = Union[DataFileSetInput, Sequence[DataFileSetInput]]


@dataclass(frozen=True, slots=True)
class DataFileSet:
    """A deterministic set of input files on a single fsspec filesystem.

    Notes:
        - This object is the resolved result: it only holds `(fs, files)`.
        - File paths are stored in the form expected by `fs.open/fs.exists` (no protocol required).
    """

    fs: AbstractFileSystem
    files: tuple[str, ...]
    _sizes: dict[str, int] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def open(self, path: str, mode: str = "rb", **kwargs):
        """Open a file path using this fileset's filesystem."""
        return self.fs.open(path, mode=mode, **kwargs)

    @property
    def is_local(self) -> bool:
        return isinstance(self.fs, LocalFileSystem)

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

        resolved_fs: AbstractFileSystem | None = fs
        collected: list[str] = []
        sizes: dict[str, int] = {}

        def _check_fs(next_fs: AbstractFileSystem) -> None:
            nonlocal resolved_fs
            if resolved_fs is None:
                resolved_fs = next_fs
                return
            if (
                type(next_fs) is not type(resolved_fs)
                or next_fs.protocol != resolved_fs.protocol  # type: ignore[attr-defined]
            ):
                raise ValueError(
                    "All inputs must resolve to the same fsspec filesystem/protocol"
                )

        def _add_files(paths: Iterable[str]) -> None:
            assert resolved_fs is not None
            for p in paths:
                if exts and not p.lower().endswith(exts):
                    continue
                if resolved_fs.exists(p) and not resolved_fs.isdir(p):
                    collected.append(p)

        def _list_dir(dir_path: str) -> Iterable[str]:
            assert resolved_fs is not None
            if recursive:
                # fsspec find() returns file paths (best-effort across backends)
                return resolved_fs.find(dir_path)

            entries = resolved_fs.ls(dir_path, detail=True)
            files: list[str] = []
            for e in entries:
                if isinstance(e, dict):
                    name = e.get("name")
                    typ = e.get("type")
                    if isinstance(name, str) and typ == "file":
                        files.append(name)
                        sz = e.get("size")
                        if isinstance(sz, int) and sz >= 0:
                            sizes.setdefault(name, sz)
                elif isinstance(e, str):
                    files.append(e)
            return files

        for item in inputs:
            if isinstance(item, DataFileSet):
                _check_fs(item.fs)
                _add_files(item.files)
                continue

            if isinstance(item, DataFile):
                _check_fs(item.fs)
                _add_files([item.path])
                continue

            if isinstance(item, DataFolder):
                # Treat as a directory spec.
                _check_fs(item.fs)
                _add_files(_list_dir(item.path))  # type: ignore[attr-defined]
                continue

            if isinstance(item, PathLike):
                item = str(item)

            if not isinstance(item, str):
                raise TypeError(
                    "DataFileSet inputs must be str | PathLike | DataFile | DataFolder | DataFileSet"
                )

            if resolved_fs is not None:
                p = resolved_fs._strip_protocol(item)  # type: ignore[attr-defined]
            else:
                next_fs, p = url_to_fs(item, **storage_options_d)
                _check_fs(next_fs)

            assert resolved_fs is not None
            if glob.has_magic(p):
                _add_files(resolved_fs.glob(p))
                continue

            if resolved_fs.exists(p):
                if resolved_fs.isdir(p):
                    _add_files(_list_dir(p))
                else:
                    _add_files([p])
                continue

            raise FileNotFoundError(f"Could not resolve input: {item!r}")

        # Deterministic sort + de-dup
        files = sorted(set(collected))

        assert resolved_fs is not None
        out = cls(fs=resolved_fs, files=tuple(files))
        # Best-effort: sizes are only available for some backends and some discovery modes.
        out._sizes.update(sizes)
        return out

    def size(self, path: str) -> int:
        """Return the size for a known file path, caching results.

        - Uses sizes discovered during `resolve()` when available.
        - Falls back to `fs.size(path)` otherwise.
        """
        if path in self._sizes:
            return self._sizes[path]
        sz = int(self.fs.size(path))
        self._sizes[path] = sz
        return sz


__all__ = ["DataFileSet", "DataFileSetLike"]
