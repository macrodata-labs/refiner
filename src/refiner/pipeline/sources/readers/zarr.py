from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal

import pyarrow as pa

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.io.zarr import zarr_store
from refiner.pipeline.data.datatype import (
    DTypeMapping,
    dtype_to_plan,
    schema_with_dtypes,
)
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.shard import RowRangeDescriptor, Shard
from refiner.pipeline.sources.base import BaseSource, SourceUnit
from refiner.pipeline.sources.readers.selection import (
    PathSelection,
    path_selection_map,
)
from refiner.utils import check_required_dependencies

ZarrMissingPolicy = Literal["error", "set_null"]


def _decode_value(value: Any) -> Any:
    if hasattr(value, "shape") and value.shape == ():
        return _decode_value(value.item())
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    return value


class ZarrReader(BaseSource):
    """Read one Zarr group as one row, or split arrays by cumulative row ends."""

    name = "read_zarr"

    def __init__(
        self,
        input: DataFolderLike,
        *,
        arrays: PathSelection | None = None,
        attrs: PathSelection | None = None,
        row_ends: str | None = None,
        rows_per_shard: int = 128,
        row_index_column: str | None = "row_index",
        file_path_column: str | None = "file_path",
        missing_policy: ZarrMissingPolicy = "error",
        dtypes: DTypeMapping | None = None,
    ):
        self.root = DataFolder.resolve(input)
        check_required_dependencies("read_zarr", ["zarr"], dist="zarr")
        self.arrays = (
            None
            if arrays is None
            else path_selection_map(
                arrays,
                format_name="Zarr",
            )
        )
        self.attrs = (
            None
            if attrs is None
            else path_selection_map(
                attrs,
                format_name="Zarr",
            )
        )
        self.row_ends = row_ends
        self.rows_per_shard = rows_per_shard
        self.row_index_column = row_index_column
        self.file_path_column = file_path_column
        self.missing_policy = missing_policy
        self.dtypes = dtypes
        _validate_output_names(
            self.arrays or {},
            self.attrs or {},
            reserved=self._reserved_output_names(row_index=row_ends is not None),
        )
        if missing_policy not in ("error", "set_null"):
            raise ValueError("missing_policy must be one of 'error' or 'set_null'")
        if (
            row_ends is not None
            and file_path_column is not None
            and file_path_column == row_index_column
        ):
            raise ValueError("file_path_column and row_index_column must be distinct")

    @property
    def schema(self) -> pa.Schema | None:
        return schema_with_dtypes(None, self.dtypes)

    def describe(self) -> dict[str, Any]:
        return {
            "path": self.root.abs_path(),
            "arrays": dict(self.arrays) if self.arrays is not None else None,
            "attrs": dict(self.attrs) if self.attrs is not None else None,
            "row_ends": self.row_ends,
            "rows_per_shard": self.rows_per_shard,
            "row_index_column": self.row_index_column,
            "file_path_column": self.file_path_column,
            "missing_policy": self.missing_policy,
            "dtypes": (
                {key: dtype_to_plan(dtype) for key, dtype in self.dtypes.items()}
                if self.dtypes
                else None
            ),
        }

    def list_shards(self) -> list[Shard]:
        path = self.root.abs_path()
        if self.row_ends is not None:
            if self.rows_per_shard <= 0:
                raise ValueError("rows_per_shard must be greater than zero")
            import zarr

            group = zarr.open_group(store=zarr_store(self.root), mode="r")
            try:
                row_count = len(group[self.row_ends])
            except KeyError:
                raise KeyError(
                    f"Zarr row_ends array not found: {self.row_ends}"
                ) from None
            return [
                Shard.from_row_range(
                    start=start,
                    end=min(start + self.rows_per_shard, row_count),
                    global_ordinal=index,
                    start_key=path,
                    end_key=path,
                )
                for index, start in enumerate(range(0, row_count, self.rows_per_shard))
            ]
        return [
            Shard.from_row_range(
                start=0,
                end=1,
                global_ordinal=0,
                start_key=path,
                end_key=path,
            )
        ]

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        import zarr

        group = zarr.open_group(store=zarr_store(self.root), mode="r")
        arrays = (
            {path: path for path in _iter_array_paths(group) if path != self.row_ends}
            if self.arrays is None
            else self.arrays
        )
        _validate_output_names(
            arrays,
            self.attrs or {},
            reserved=self._reserved_output_names(row_index=self.row_ends is not None),
        )
        if self.row_ends is not None:
            descriptor = shard.descriptor
            assert isinstance(descriptor, RowRangeDescriptor)
            try:
                ends_array = group[self.row_ends]
                row_ends = [
                    int(value)
                    for value in ends_array[descriptor.start : descriptor.end]
                ]
                start = (
                    0
                    if descriptor.start == 0
                    else int(ends_array[descriptor.start - 1])
                )
            except KeyError:
                raise KeyError(
                    f"Zarr row_ends array not found: {self.row_ends}"
                ) from None
            shard_start = start
            shard_end = row_ends[-1] if row_ends else start
            shard_arrays = self._read_arrays(
                group,
                arrays,
                start=shard_start,
                end=shard_end,
            )
            if shard_arrays is None:
                return
            for offset, end in enumerate(row_ends):
                row = self._row_metadata(row_index=descriptor.start + offset)
                relative_start = start - shard_start
                relative_end = end - shard_start
                for output_name, value in shard_arrays.items():
                    row[output_name] = (
                        None if value is None else value[relative_start:relative_end]
                    )
                row = self._read_attrs(group, row)
                if row is None:
                    return
                yield DictRow(row)
                start = end
            return

        row = self._row_metadata(row_index=None)
        row_arrays = self._read_arrays(group, arrays)
        if row_arrays is None:
            return
        row.update(row_arrays)
        row = self._read_attrs(group, row)
        if row is not None:
            yield DictRow(row)

    def _reserved_output_names(self, *, row_index: bool) -> set[str]:
        names = set()
        if self.file_path_column is not None:
            names.add(self.file_path_column)
        if row_index and self.row_index_column is not None:
            names.add(self.row_index_column)
        return names

    def _row_metadata(self, *, row_index: int | None) -> dict[str, Any]:
        row: dict[str, Any] = {}
        if self.file_path_column is not None:
            row[self.file_path_column] = self.root.abs_path()
        if self.row_index_column is not None and row_index is not None:
            row[self.row_index_column] = row_index
        return row

    def _read_arrays(
        self,
        group: Any,
        arrays: Mapping[str, str],
        *,
        start: int | None = None,
        end: int | None = None,
    ) -> dict[str, Any] | None:
        row: dict[str, Any] = {}
        for output_name, path in arrays.items():
            try:
                row[output_name] = (
                    group[path][start:end] if start is not None else group[path][:]
                )
            except KeyError:
                if self.missing_policy == "set_null":
                    row[output_name] = None
                    continue
                raise KeyError(f"Zarr array not found: {path}") from None
        return row

    def _read_attrs(self, group: Any, row: dict[str, Any]) -> dict[str, Any] | None:
        for output_name, attr_name in (self.attrs or {}).items():
            if attr_name not in group.attrs:
                if self.missing_policy == "set_null":
                    row[output_name] = None
                    continue
                raise KeyError(f"Zarr attr not found: {attr_name}")
            row[output_name] = _decode_value(group.attrs[attr_name])
        return row


def _validate_output_names(
    arrays: Mapping[str, str],
    attrs: Mapping[str, str],
    *,
    reserved: set[str] | None = None,
) -> None:
    duplicates = set(arrays).intersection(attrs)
    if duplicates:
        names = ", ".join(sorted(repr(name) for name in duplicates))
        raise ValueError(f"Zarr arrays and attrs use duplicate output names: {names}")
    reserved_matches = set(arrays).union(attrs).intersection(reserved or set())
    if reserved_matches:
        names = ", ".join(sorted(repr(name) for name in reserved_matches))
        raise ValueError(f"Zarr selections use reserved output names: {names}")


def _iter_array_paths(group: Any, prefix: str = "") -> Iterator[str]:
    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if hasattr(item, "shape"):
            yield path
        else:
            yield from _iter_array_paths(item, path)


__all__ = ["PathSelection", "ZarrMissingPolicy", "ZarrReader"]
