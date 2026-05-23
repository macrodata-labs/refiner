from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal, cast

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
from refiner.utils import check_required_dependencies

MissingPolicy = Literal["error", "drop_row", "set_null"]
PathSelection = Mapping[str, str] | Sequence[str] | str


def _selection_map(value: PathSelection) -> dict[str, str]:
    if isinstance(value, str):
        return {value.rsplit("/", 1)[-1]: value}
    if isinstance(value, Mapping):
        return dict(cast(Mapping[str, str], value))
    out: dict[str, str] = {}
    for path in value:
        name = path.rsplit("/", 1)[-1]
        if name in out:
            raise ValueError(
                "Zarr path selections must have unique derived column names; "
                f"use an explicit mapping for duplicate name {name!r}"
            )
        out[name] = path
    return out


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
        missing_policy: MissingPolicy = "error",
        dtypes: DTypeMapping | None = None,
    ):
        self.root = DataFolder.resolve(input)
        check_required_dependencies("read_zarr", ["zarr"], dist="zarr")
        self.arrays = None if arrays is None else _selection_map(arrays)
        self.attrs = None if attrs is None else _selection_map(attrs)
        _validate_output_names(self.arrays or {}, self.attrs or {})
        self.row_ends = row_ends
        self.rows_per_shard = rows_per_shard
        self.row_index_column = row_index_column
        self.file_path_column = file_path_column
        self.missing_policy = missing_policy
        self.dtypes = dtypes
        if missing_policy not in ("error", "drop_row", "set_null"):
            raise ValueError(
                "missing_policy must be one of 'error', 'drop_row', or 'set_null'"
            )

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
            row_count = len(group[self.row_ends])
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
                if self.missing_policy == "drop_row":
                    return
                raise KeyError(
                    f"Zarr row_ends array not found: {self.row_ends}"
                ) from None
            for offset, end in enumerate(row_ends):
                row = self._read_row(
                    group,
                    arrays,
                    start=start,
                    end=end,
                    row_index=descriptor.start + offset,
                )
                if row is None:
                    return
                yield DictRow(row)
                start = end
            return

        row = self._read_row(group, arrays)
        if row is not None:
            yield DictRow(row)

    def _read_row(
        self,
        group: Any,
        arrays: Mapping[str, str],
        *,
        start: int | None = None,
        end: int | None = None,
        row_index: int | None = None,
    ) -> dict[str, Any] | None:
        row: dict[str, Any] = {}
        if self.file_path_column is not None:
            row[self.file_path_column] = self.root.abs_path()
        if self.row_index_column is not None and row_index is not None:
            row[self.row_index_column] = row_index
        for output_name, path in arrays.items():
            try:
                row[output_name] = (
                    group[path][start:end] if start is not None else group[path][:]
                )
            except KeyError:
                if self.missing_policy == "drop_row":
                    return None
                if self.missing_policy == "set_null":
                    row[output_name] = None
                    continue
                raise KeyError(f"Zarr array not found: {path}") from None
        for output_name, attr_name in (self.attrs or {}).items():
            if attr_name not in group.attrs:
                if self.missing_policy == "drop_row":
                    return None
                if self.missing_policy == "set_null":
                    row[output_name] = None
                    continue
                raise KeyError(f"Zarr attr not found: {attr_name}")
            row[output_name] = _decode_value(group.attrs[attr_name])
        return row


def _validate_output_names(
    arrays: Mapping[str, str],
    attrs: Mapping[str, str],
) -> None:
    duplicates = set(arrays).intersection(attrs)
    if duplicates:
        names = ", ".join(sorted(repr(name) for name in duplicates))
        raise ValueError(f"Zarr arrays and attrs use duplicate output names: {names}")


def _iter_array_paths(group: Any, prefix: str = "") -> Iterator[str]:
    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if hasattr(item, "shape"):
            yield path
        else:
            yield from _iter_array_paths(item, path)


__all__ = ["MissingPolicy", "PathSelection", "ZarrReader"]
