from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from fsspec import AbstractFileSystem

from refiner.io import DataFileSet
from refiner.io.fileset import DataFileSetLike
from refiner.ledger.shard import Shard
from refiner.sources.base import BaseSource


class BaseReader(BaseSource):
    """Base class for file-backed readers.

    Responsibilities:
        - Resolve an input fsspec path into a deterministic list of file paths.
        - Provide shard listing and shard reading.

    Note:
        This object is expected to be used by a single worker at a time (no concurrent read_shard calls).
    """

    name: str = ""

    def __init__(
        self,
        inputs: DataFileSetLike,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        recursive: bool = False,
        extensions: Sequence[str] = (),
    ):
        """Create a reader over a set of input files.

        Args:
            inputs: Input spec(s): paths, globs, folders, or `DataFile`/`DataFolder`/`DataFileSet`.
            fs: Optional initialized filesystem to use for string inputs.
            storage_options: Optional fsspec init options (used only when `fs` is not provided).
            recursive: If a directory input is provided, whether to list recursively.
            extensions: If a directory input is provided, filter by these suffixes (case-insensitive).
        """
        self._inputs = inputs
        self._fs = fs
        self._storage_options = storage_options
        self._recursive = recursive
        self._extensions = tuple(extensions)
        self._fileset: DataFileSet | None = None
        # Single-open-file cache for readers that do byte-based seeks or stream reads.
        self._open_path: str | None = None
        self._open_fh: Any | None = None

        if not self.name:
            reader_name = self.__class__.__name__.replace("Reader", "").lower()
            self.name = f"read_{reader_name}"

    @property
    def fileset(self) -> DataFileSet:
        """Resolved input files and filesystem (cached)."""
        if self._fileset is None:
            self._fileset = DataFileSet.resolve(
                self._inputs,
                fs=self._fs,
                storage_options=self._storage_options,
                recursive=self._recursive,
                extensions=self._extensions,
            )
        return self._fileset

    @property
    def fs(self) -> AbstractFileSystem:
        """Filesystem used to open/read input files."""
        return self.fileset.fs

    @property
    def files(self) -> list[str]:
        """Deterministic list of resolved input file paths (fs-native, protocol-stripped)."""
        return list(self.fileset.files)

    def describe(self) -> dict[str, Any]:
        # Keep planning metadata cheap: do not resolve/list inputs here.
        raw = self._inputs
        if isinstance(raw, (str, Path)):
            return {"path": str(raw)}
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if not raw:
                return {}
            first = raw[0]
            if isinstance(first, (str, Path)):
                if len(raw) == 1:
                    return {"path": str(first)}
                return {"path": f"{first} (+{len(raw) - 1} more)"}
        return {}

    def _get_file_handle(
        self, path: str, *, mode: str = "rb", force_reopen: bool = False
    ):
        """Get a cached fsspec file handle for `path`

        Returns:
            (fh, opened_new): `opened_new` is True if a new file handle was opened.
        """
        if not force_reopen and self._open_path == path and self._open_fh is not None:
            return self._open_fh, False

        if self._open_fh is not None:
            try:
                self._open_fh.close()
            except Exception:
                pass
            self._open_fh = None
            self._open_path = None

        self._open_fh = self.fs.open(path, mode=mode)
        self._open_path = path
        return self._open_fh, True

    @abstractmethod
    def list_shards(self) -> list[Shard]:
        """Return the deterministic list of shards for this reader.

        Contract:
            - Shards must only reference paths in `self.files`.
            - Shards must not span multiple files.
            - `start/end` are reader-specific offsets (e.g. bytes or row-group indices).
        """
        raise NotImplementedError

    @abstractmethod
    def read_shard(self, shard: Shard) -> Iterator[Any]:
        """Read a shard and yield row units.

        Contract:
            - Must accept shards returned by `list_shards()`.
            - Should be safe to call sequentially (single-worker, no concurrent calls).
            - Units can be `Row` or Arrow tabular blocks (`RecordBatch`/`Table`).
        """
        raise NotImplementedError


__all__ = ["BaseReader"]
