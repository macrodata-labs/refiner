from __future__ import annotations

import glob
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, TypeAlias, Union, cast

from fsspec import AbstractFileSystem, url_to_fs

from refiner.io.datafile import DataFile, DataFileSpec
from refiner.io.datafolder import DataFolder, DataFolderSpec

DataFileSetInput: TypeAlias = Union[
    str, PathLike[str], DataFileSpec, DataFolderSpec, DataFile, DataFolder
]
DataFileSetLike: TypeAlias = Union[DataFileSetInput, Sequence[DataFileSetInput]]


@dataclass(frozen=True, slots=True)
class _PathSource:
    path: str
    fs: AbstractFileSystem


@dataclass(frozen=True, slots=True)
class DataFileSet:
    """A deterministic set of normalized input sources.

    Notes:
        - This object preserves user input order without eagerly listing or globbing.
        - Source entries are normalized to `DataFile`, `DataFolder`, or a deferred string path.
        - `(path, fs)` inputs are accepted and kept lazy like plain string paths.
        - Nested `DataFileSet` inputs are not supported.
        - Concrete files are expanded lazily when `files`, `expand_sources()`, or `size()` is used.
    """

    entries: tuple[DataFile | DataFolder | _PathSource, ...]
    recursive: bool = False
    extensions: tuple[str, ...] = ()
    _expanded_sources: tuple[tuple[DataFile, ...], ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _sizes: dict[tuple[int, str], int] = field(
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
        """Normalize input specs into a lazy file set without listing them yet."""
        if isinstance(data, cls):
            return data

        if (
            isinstance(data, tuple)
            and len(data) == 2
            and isinstance(data[1], AbstractFileSystem)
        ):
            inputs = (data,)
        elif isinstance(data, (str, PathLike, DataFile, DataFolder)):
            inputs = (data,)
        else:
            inputs = tuple(data)

        normalized_entries: list[DataFile | DataFolder | _PathSource] = []

        for item in inputs:
            if isinstance(item, DataFile):
                normalized_entries.append(item)
                continue

            if isinstance(item, DataFolder):
                normalized_entries.append(item)
                continue

            if (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[1], AbstractFileSystem)
            ):
                path, item_fs = cast(DataFileSpec | DataFolderSpec, item)
                if isinstance(path, PathLike):
                    path = str(path)
                if not isinstance(path, str):
                    raise TypeError(
                        "DataFileSet inputs must be str | PathLike | (path, fs) | DataFile | DataFolder"
                    )
                normalized_entries.append(
                    _PathSource(path=item_fs._strip_protocol(path), fs=item_fs)
                )
                continue

            if isinstance(item, PathLike):
                item = str(item)

            if not isinstance(item, str):
                raise TypeError(
                    "DataFileSet inputs must be str | PathLike | (path, fs) | DataFile | DataFolder"
                )

            if fs is not None:
                normalized_entries.append(
                    _PathSource(path=fs._strip_protocol(item), fs=fs)
                )
            else:
                item_fs, path = url_to_fs(item, **dict(storage_options or {}))
                normalized_entries.append(_PathSource(path=path, fs=item_fs))

        return cls(
            entries=tuple(normalized_entries),
            recursive=recursive,
            extensions=tuple(extensions or ()),
        )

    @property
    def files(self) -> tuple[DataFile, ...]:
        """Flatten the lazily expanded source groups into one deterministic file list."""
        return tuple(file for group in self.expand_sources() for file in group)

    def expand_sources(self) -> tuple[tuple[DataFile, ...], ...]:
        """Expand each source entry into its concrete files, preserving source grouping."""
        cached = self._expanded_sources
        if cached is not None:
            return cached

        exts = tuple(e.lower() for e in self.extensions)
        seen: set[tuple[int, str]] = set()
        expanded: list[tuple[DataFile, ...]] = []

        def _append_file(out: list[DataFile], file: DataFile) -> None:
            if exts and not file.path.lower().endswith(exts):
                return
            key = (id(file.fs), file.path)
            if key in seen:
                return
            seen.add(key)
            out.append(file)

        for entry in self.entries:
            files: list[DataFile] = []
            if isinstance(entry, DataFile):
                _append_file(files, entry)
            elif isinstance(entry, DataFolder):
                paths = (
                    sorted(entry.find(""))
                    if self.recursive
                    else sorted(
                        e["name"] if isinstance(e, dict) else e
                        for e in entry.ls("", detail=True)
                        if not isinstance(e, dict) or e.get("type") == "file"
                    )
                )
                for path in paths:
                    _append_file(files, entry.file(path))
            else:
                next_fs, path = entry.fs, entry.path
                if glob.has_magic(path):
                    for expanded_path in sorted(next_fs.glob(path)):
                        _append_file(files, DataFile(fs=next_fs, path=expanded_path))
                elif next_fs.exists(path):
                    if next_fs.isdir(path):
                        paths = (
                            sorted(next_fs.find(path))
                            if self.recursive
                            else sorted(
                                e["name"] if isinstance(e, dict) else e
                                for e in next_fs.ls(path, detail=True)
                                if not isinstance(e, dict) or e.get("type") == "file"
                            )
                        )
                        for expanded_path in paths:
                            _append_file(
                                files, DataFile(fs=next_fs, path=expanded_path)
                            )
                    else:
                        _append_file(files, DataFile(fs=next_fs, path=path))
                else:
                    raise FileNotFoundError(
                        f"Could not resolve input: {next_fs.unstrip_protocol(path)!r}"
                    )
            expanded.append(tuple(files))

        out = tuple(expanded)
        object.__setattr__(self, "_expanded_sources", out)
        return out

    def resolve_file(self, source_index: int, path: str) -> DataFile:
        """Resolve an absolute shard path back onto the source entry's filesystem."""
        entry = self.entries[source_index]
        if isinstance(entry, DataFile):
            file = DataFile.resolve(path, fs=entry.fs)
            if (
                file.path != entry.path
                and str(entry) != path
                and entry.abs_path() != path
            ):
                raise FileNotFoundError(path)
            return file
        if isinstance(entry, DataFolder):
            return DataFile.resolve(path, fs=entry.fs)
        return DataFile.resolve(path, fs=entry.fs)

    def size(self, source_index: int, path: str) -> int:
        """Return the size for a resolved shard path, caching by source entry and absolute path."""
        key = (source_index, path)
        if key in self._sizes:
            return self._sizes[key]
        target = self.resolve_file(source_index, path)
        sz = int(target.fs.size(target.path))
        self._sizes[key] = sz
        return sz


__all__ = ["DataFileSet", "DataFileSetLike"]
