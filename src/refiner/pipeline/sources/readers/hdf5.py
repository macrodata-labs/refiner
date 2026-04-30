from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import fnmatch
from glob import has_magic
from typing import Any, Literal, cast

from fsspec import AbstractFileSystem

from refiner.io import DataFile
from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.datatype import DTypeMapping, dtype_to_plan
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.shard import FilePartsDescriptor
from refiner.pipeline.sources.readers.base import BaseReader, Shard, SourceUnit
from refiner.pipeline.sources.readers.utils import DEFAULT_TARGET_SHARD_BYTES
from refiner.utils import check_required_dependencies


MissingPolicy = Literal["error", "drop_row", "set_null"]
PathSelection = Mapping[str, str] | Sequence[str] | str


def _decode_value(
    value: Any,
    *,
    decode_bytes: bool = True,
    preserve_arrays: bool = False,
) -> Any:
    if isinstance(value, bytes):
        if not decode_bytes:
            return value
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    if isinstance(value, str) and any("\udc80" <= char <= "\udcff" for char in value):
        return value.encode("utf-8", errors="surrogateescape")
    if hasattr(value, "shape") and value.shape == ():
        return _decode_value(
            value.item(),
            decode_bytes=decode_bytes,
            preserve_arrays=preserve_arrays,
        )
    if hasattr(value, "tolist"):
        if preserve_arrays and getattr(
            getattr(value, "dtype", None), "kind", None
        ) not in (
            "O",
            "S",
        ):
            return value
        return _decode_value(
            value.tolist(),
            decode_bytes=decode_bytes,
            preserve_arrays=preserve_arrays,
        )
    if isinstance(value, list):
        return [
            _decode_value(
                item,
                decode_bytes=decode_bytes,
                preserve_arrays=preserve_arrays,
            )
            for item in value
        ]
    return value


class Hdf5Reader(BaseReader):
    """HDF5 reader planned at file granularity.

    Each matched HDF5 group becomes one output row. Shard planning only resolves
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
        datasets: PathSelection | None = None,
        attrs: PathSelection | None = None,
        file_path_column: str | None = "file_path",
        group_path_column: str | None = "hdf5_group",
        missing_policy: MissingPolicy = "error",
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
            dtypes=dtypes,
        )
        if groups is None:
            self.groups = ("/",)
        elif isinstance(groups, str):
            group = "/" if groups in ("", "/") else groups
            group = group if group.startswith("/") else f"/{group}"
            if sum(part == "**" for part in group.split("/")) > 1:
                raise ValueError("groups glob may contain at most one '**' segment")
            self.groups = group if has_magic(group) else (group,)
        else:
            self.groups = tuple(
                "/"
                if group in ("", "/")
                else group
                if group.startswith("/")
                else f"/{group}"
                for group in groups
            )
            if any(has_magic(group) for group in self.groups):
                raise ValueError(
                    "groups accepts a single glob string or a list of exact group paths"
                )
        self.datasets = self._mapping(datasets)
        self.attrs = self._mapping(attrs)
        self.group_path_column = group_path_column
        self.missing_policy = missing_policy
        if missing_policy not in ("error", "drop_row", "set_null"):
            raise ValueError(
                "missing_policy must be one of 'error', 'drop_row', or 'set_null'"
            )
        self._validate_column_names()

    @staticmethod
    def _mapping(
        value: PathSelection | None,
    ) -> dict[str, str]:
        if value is None:
            return {}
        if isinstance(value, str):
            return {value.rsplit("/", 1)[-1]: value}
        if isinstance(value, Mapping):
            return dict(cast(Mapping[str, str], value))
        out: dict[str, str] = {}
        for path in value:
            name = path.rsplit("/", 1)[-1]
            if name in out:
                raise ValueError(
                    "HDF5 path selections must have unique derived column names; "
                    f"use an explicit mapping for duplicate name {name!r}"
                )
            out[name] = path
        return out

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "groups": [self.groups]
                if isinstance(self.groups, str)
                else list(self.groups),
                "datasets": dict(self.datasets),
                "attrs": dict(self.attrs),
                "group_path_column": self.group_path_column,
                "missing_policy": self.missing_policy,
                "dtypes": (
                    {key: dtype_to_plan(dtype) for key, dtype in self.dtypes.items()}
                    if self.dtypes
                    else None
                ),
            }
        )
        return description

    def _validate_column_names(self) -> None:
        for name, path in self.datasets.items():
            if path.startswith("/"):
                raise ValueError(
                    "HDF5 dataset selections must be relative to each matched group; "
                    f"column {name!r} uses absolute path {path!r}"
                )

        dataset_names = set(self.datasets)
        attr_names = set(self.attrs)
        selected_names = dataset_names | attr_names

        overlaps = dataset_names & attr_names
        if overlaps:
            name = sorted(overlaps)[0]
            raise ValueError(
                "HDF5 dataset and attr selections must use distinct output names; "
                f"duplicate name {name!r}"
            )

        reserved = {
            name
            for name in (self.file_path_column, self.group_path_column)
            if name is not None
        }
        if len(reserved) < sum(
            name is not None for name in (self.file_path_column, self.group_path_column)
        ):
            raise ValueError("file_path_column and group_path_column must be distinct")

        collisions = selected_names & reserved
        if collisions:
            name = sorted(collisions)[0]
            raise ValueError(
                "HDF5 selected output names must not collide with metadata columns; "
                f"duplicate name {name!r}"
            )

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)
        check_required_dependencies("read_hdf5", ["h5py"], dist="hdf5")
        import h5py

        for part in descriptor.parts:
            source = self.fileset.resolve_file(part.source_index, part.path)
            with source.open(mode="rb") as raw, h5py.File(raw, "r") as h5:
                for group_path, group in self._iter_groups(h5, h5py):
                    row = self._read_group(source, group, group_path, h5py)
                    if row is not None:
                        yield DictRow(row)

    def _iter_groups(self, h5, h5py) -> Iterator[tuple[str, Any]]:
        if isinstance(self.groups, str):
            yield from self._expand_group_glob(h5, self.groups, h5py)
            return

        for group_path in self.groups:
            if group_path == "/":
                yield group_path, h5
            elif group_path in h5 and isinstance(h5[group_path], h5py.Group):
                yield group_path, h5[group_path]

    def _expand_group_glob(self, h5, pattern: str, h5py) -> Iterator[tuple[str, Any]]:
        visited_recursive_groups: set[Any] = set()

        def visit(group, path: str, parts: Sequence[str]) -> Iterator[tuple[str, Any]]:
            if not parts:
                yield path, group
                return

            part = parts[0]
            rest = parts[1:]
            if part == "**":
                if group.id in visited_recursive_groups:
                    return
                visited_recursive_groups.add(group.id)
                yield from visit(group, path, rest)
                for name, obj in group.items():
                    if isinstance(obj, h5py.Group):
                        child_path = f"/{name}" if path == "/" else f"{path}/{name}"
                        yield from visit(obj, child_path, parts)
                return

            if has_magic(part):
                for name, obj in group.items():
                    if fnmatch.fnmatchcase(name, part) and isinstance(obj, h5py.Group):
                        child_path = f"/{name}" if path == "/" else f"{path}/{name}"
                        yield from visit(obj, child_path, rest)
                return

            if part in group and isinstance(group[part], h5py.Group):
                child_path = f"/{part}" if path == "/" else f"{path}/{part}"
                yield from visit(group[part], child_path, rest)

        yield from visit(h5, "/", [part for part in pattern.split("/") if part])

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
                if self.missing_policy == "drop_row":
                    return None
                if self.missing_policy == "set_null":
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
            row[output_name] = _decode_value(
                dataset[()],
                decode_bytes=dataset.dtype.kind != "S",
                preserve_arrays=True,
            )

        for output_name, attr_name in self.attrs.items():
            if attr_name not in group.attrs:
                if self.missing_policy == "drop_row":
                    return None
                if self.missing_policy == "set_null":
                    row[output_name] = None
                    continue
                raise KeyError(f"HDF5 attr not found on {group_path}: {attr_name}")
            row[output_name] = _decode_value(group.attrs[attr_name])

        return self._with_file_path(row, source)


__all__ = ["Hdf5Reader"]
