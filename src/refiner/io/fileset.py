from __future__ import annotations

import glob
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Literal, TypeAlias, Union, cast

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
        expect_type: Literal["file", "folder"] | None = None,
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
                if expect_type == "folder":
                    raise TypeError("inputs must be directories")
                normalized_entries.append(item)
                continue

            if isinstance(item, DataFolder):
                if expect_type == "file":
                    raise TypeError("inputs must be files")
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
                if expect_type == "folder":
                    normalized_entries.append(DataFolder(path=path, fs=item_fs))
                elif expect_type == "file":
                    normalized_entries.append(DataFile(path=path, fs=item_fs))
                else:
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
                if expect_type == "folder":
                    normalized_entries.append(DataFolder.resolve(item, fs=fs))
                elif expect_type == "file":
                    normalized_entries.append(DataFile.resolve(item, fs=fs))
                else:
                    normalized_entries.append(
                        _PathSource(path=fs._strip_protocol(item), fs=fs)
                    )
            else:
                if expect_type == "folder":
                    normalized_entries.append(
                        DataFolder.resolve(item, storage_options=storage_options)
                    )
                elif expect_type == "file":
                    normalized_entries.append(
                        DataFile.resolve(item, storage_options=storage_options)
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

    @property
    def datafiles(self) -> tuple[DataFile, ...]:
        if not all(isinstance(entry, DataFile) for entry in self.entries):
            raise TypeError("DataFileSet entries are not all files")
        return cast(tuple[DataFile, ...], self.entries)

    @property
    def datafolders(self) -> tuple[DataFolder, ...]:
        if not all(isinstance(entry, DataFolder) for entry in self.entries):
            raise TypeError("DataFileSet entries are not all folders")
        return cast(tuple[DataFolder, ...], self.entries)

    def expand_sources(self) -> tuple[tuple[DataFile, ...], ...]:
        """Expand each source entry into its concrete files, preserving source grouping."""
        cached = self._expanded_sources
        if cached is not None:
            return cached

        exts = tuple(e.lower() for e in self.extensions)
        seen: set[tuple[int, str]] = set()
        expanded: list[tuple[DataFile, ...]] = []
        sizes = dict(self._sizes)

        def _append_file(
            out: list[DataFile],
            file: DataFile,
            *,
            size: int | None = None,
            apply_extensions: bool = True,
        ) -> None:
            if apply_extensions and exts and not file.path.lower().endswith(exts):
                return
            key = (id(file.fs), file.path)
            if key in seen:
                return
            seen.add(key)
            out.append(file)
            if size is not None:
                sizes[(len(expanded), file.abs_path())] = int(size)

        for entry in self.entries:
            files: list[DataFile] = []
            if isinstance(entry, _PathSource) and not glob.has_magic(entry.path):
                try:
                    info = entry.fs.info(entry.path)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Could not resolve input: {entry.fs.unstrip_protocol(entry.path)!r}"
                    )
                item_type = info.get("type")
                if item_type == "directory":
                    entry = DataFolder(path=entry.path, fs=entry.fs)
                elif item_type == "file":
                    entry = DataFile(fs=entry.fs, path=entry.path)
                else:
                    raise TypeError(
                        f"Unsupported file type {item_type!r} for input: "
                        f"{entry.fs.unstrip_protocol(entry.path)!r}"
                    )

            if isinstance(entry, DataFile):
                _append_file(files, entry, apply_extensions=False)
            elif isinstance(entry, DataFolder):
                for file, size in entry.iter_files_with_sizes(recursive=self.recursive):
                    _append_file(files, file, size=size)
            else:
                next_fs, path = entry.fs, entry.path
                if glob.has_magic(path):
                    matched = next_fs.glob(path, detail=True)
                    items = matched.items()
                    for expanded_path, info in sorted(items):
                        if not isinstance(expanded_path, str) or not isinstance(
                            info, Mapping
                        ):
                            continue
                        if info.get("type") != "file":
                            continue
                        size = info.get("size")
                        _append_file(
                            files,
                            DataFile(fs=next_fs, path=expanded_path),
                            size=size if isinstance(size, int) else None,
                        )
                else:
                    raise AssertionError(
                        "non-glob _PathSource should have been resolved"
                    )
            expanded.append(tuple(files))

        out = tuple(expanded)
        object.__setattr__(self, "_expanded_sources", out)
        object.__setattr__(self, "_sizes", sizes)
        return out

    def resolve_file(self, source_index: int, path: str) -> DataFile:
        """Resolve an absolute shard path back onto the source entry's filesystem."""
        entry = self.entries[source_index]
        if isinstance(entry, DataFile):
            if path in {entry.path, str(entry), entry.abs_path()}:
                return entry
            file = DataFile.resolve(path, fs=entry.fs)
            if file.path.lstrip("/") != entry.path.lstrip("/"):
                raise FileNotFoundError(path)
            return entry
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
