from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any, Literal, cast

from fsspec import AbstractFileSystem
import pyarrow as pa

from refiner.io import DataFile
from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.datatype import DTypeMapping, schema_with_dtypes
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.shard import FilePartsDescriptor
from refiner.pipeline.sources.readers.base import BaseReader, Shard, SourceUnit
from refiner.pipeline.sources.readers.utils import DEFAULT_TARGET_SHARD_BYTES


MissingPolicy = Literal["error", "skip", "none"]


def _load_h5py():
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "read_hdf5 requires h5py. Install it with `pip install "
            "macrodata-refiner[hdf5]`."
        ) from e
    return h5py


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "shape") and value.shape == ():
        return _decode_scalar(value.item())
    return value


def _has_glob(value: str) -> bool:
    return any(char in value for char in "*?[")


class Hdf5Reader(BaseReader):
    """HDF5 reader planned at file granularity.

    Each matched HDF5 group becomes one output row. `list_shards()` only resolves
    files and sizes; it does not open HDF5 files or inspect group metadata.
    """

    name = "read_hdf5"

    def __init__(
        self,
        inputs: DataFileSetLike,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        recursive: bool = False,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
        groups: str | Sequence[str] | None = None,
        datasets: Mapping[str, str] | Sequence[str] | None = None,
        attrs: Mapping[str, str] | Sequence[str] | None = None,
        file_path_column: str | None = "file_path",
        group_path_column: str | None = "hdf5_group",
        missing: MissingPolicy = "error",
        dtypes: DTypeMapping | None = None,
    ):
        super().__init__(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            extensions=(".h5", ".hdf5"),
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            file_path_column=file_path_column,
            split_by_bytes=False,
        )
        self.groups = ("/",) if groups is None else self._tuple(groups)
        self.datasets = self._mapping(datasets)
        self.attrs = self._mapping(attrs)
        self.group_path_column = group_path_column
        self.missing = missing
        self.dtypes = dtypes
        if missing not in ("error", "skip", "none"):
            raise ValueError("missing must be one of 'error', 'skip', or 'none'")

    @staticmethod
    def _tuple(value: str | Sequence[str]) -> tuple[str, ...]:
        return (value,) if isinstance(value, str) else tuple(value)

    @staticmethod
    def _mapping(
        value: Mapping[str, str] | Sequence[str] | None,
    ) -> dict[str, str]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            typed = cast(Mapping[str, str], value)
            return {key: path for key, path in typed.items()}
        return {path.rsplit("/", 1)[-1]: path for path in value}

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "groups": list(self.groups),
                "datasets": dict(self.datasets),
                "attrs": dict(self.attrs),
                "group_path_column": self.group_path_column,
                "missing": self.missing,
                "dtypes": list(self.dtypes) if self.dtypes else None,
            }
        )
        return description

    @property
    def schema(self) -> pa.Schema | None:
        return schema_with_dtypes(None, self.dtypes)

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)
        h5py = _load_h5py()

        for part in descriptor.parts:
            source = self.fileset.resolve_file(part.source_index, part.path)
            with source.open(mode="rb") as raw, h5py.File(raw, "r") as h5:
                for group_path in self._matched_groups(h5, h5py):
                    row = self._read_group(source, h5[group_path], group_path, h5py)
                    if row is not None:
                        yield DictRow(row)

    def _matched_groups(self, h5, h5py) -> list[str]:
        matches: set[str] = set()
        for pattern in self.groups:
            if not _has_glob(pattern):
                path = "/" if pattern in ("", "/") else pattern
                if not path.startswith("/"):
                    path = f"/{path}"
                if path in h5 and isinstance(h5[path], h5py.Group):
                    matches.add(path)
                elif self.missing == "error":
                    raise KeyError(f"HDF5 group not found: {path}")
                continue

            def visit(name: str, obj) -> None:
                path = f"/{name}" if name else "/"
                if isinstance(obj, h5py.Group) and PurePosixPath(path).match(pattern):
                    matches.add(path)

            h5.visititems(visit)

        return sorted(matches)

    def _read_group(
        self,
        source: DataFile,
        group,
        group_path: str,
        h5py,
    ) -> dict[str, Any] | None:
        row: dict[str, Any] = {}
        if self.group_path_column is not None:
            row[self.group_path_column] = group_path

        for output_name, dataset_path in self.datasets.items():
            if dataset_path not in group:
                if self.missing == "skip":
                    return None
                if self.missing == "none":
                    row[output_name] = None
                    continue
                raise KeyError(
                    f"HDF5 dataset not found under {group_path}: {dataset_path}"
                )

            dataset = group[dataset_path]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(
                    f"HDF5 path under {group_path} is not a dataset: {dataset_path}"
                )
            row[output_name] = _decode_scalar(dataset[()])

        for output_name, attr_name in self.attrs.items():
            if attr_name not in group.attrs:
                if self.missing == "skip":
                    return None
                if self.missing == "none":
                    row[output_name] = None
                    continue
                raise KeyError(f"HDF5 attr not found on {group_path}: {attr_name}")
            row[output_name] = _decode_scalar(group.attrs[attr_name])

        return self._with_file_path(row, source)


__all__ = ["Hdf5Reader"]
