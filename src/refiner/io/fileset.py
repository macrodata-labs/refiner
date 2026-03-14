from __future__ import annotations

import glob
from collections.abc import Mapping, Sequence
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
class _PathSource:
    raw: str
    fs: AbstractFileSystem | None = None
    storage_options: Mapping[str, Any] | None = None

    def resolve(self) -> tuple[AbstractFileSystem, str]:
        if self.fs is not None:
            return self.fs, self.fs._strip_protocol(self.raw)
        return url_to_fs(self.raw, **dict(self.storage_options or {}))


@dataclass(frozen=True, slots=True)
class DataFileSet:
    """A deterministic set of normalized input sources.

    Notes:
        - This object preserves user input order without eagerly listing or globbing.
        - Source entries are normalized to `DataFile`, `DataFolder`, or a deferred string path.
        - Concrete files are expanded lazily when `files`, `expand_sources()`, or `size()` is used.
    """

    entries: tuple[DataFile | DataFolder | _PathSource, ...]
    recursive: bool = False
    extensions: tuple[str, ...] = ()
    _expanded_sources: tuple[tuple[DataFile, ...], ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )
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
        """Normalize input specs into a lazy file set without listing them yet."""
        if isinstance(data, cls):
            return data

        if isinstance(data, (str, PathLike, DataFile, DataFolder, DataFileSet)):
            inputs = (data,)
        else:
            inputs = tuple(data)

        storage_options_d = dict(storage_options or {})
        normalized_entries: list[DataFile | DataFolder | _PathSource] = []

        for item in inputs:
            if isinstance(item, DataFileSet):
                normalized_entries.extend(item.entries)
                continue

            if isinstance(item, DataFile):
                normalized_entries.append(item)
                continue

            if isinstance(item, DataFolder):
                normalized_entries.append(item)
                continue

            if isinstance(item, PathLike):
                item = str(item)

            if not isinstance(item, str):
                raise TypeError(
                    "DataFileSet inputs must be str | PathLike | DataFile | DataFolder | DataFileSet"
                )

            normalized_entries.append(
                _PathSource(
                    raw=item,
                    fs=fs,
                    storage_options=storage_options_d if fs is None else None,
                )
            )

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
                next_fs, path = entry.resolve()
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
                    raise FileNotFoundError(f"Could not resolve input: {entry.raw!r}")
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
        fs, _ = entry.resolve()
        return DataFile.resolve(path, fs=fs)

    def size(self, source_index: int, path: str) -> int:
        """Return the size for a resolved shard path, caching by source entry and absolute path."""
        key = hash((source_index, path))
        if key in self._sizes:
            return self._sizes[key]
        target = self.resolve_file(source_index, path)
        sz = int(target.fs.size(target.path))
        self._sizes[key] = sz
        return sz


__all__ = ["DataFileSet", "DataFileSetLike"]
