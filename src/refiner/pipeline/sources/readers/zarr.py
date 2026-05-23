from __future__ import annotations

from collections.abc import Iterator, Mapping
from math import ceil, prod
from typing import Any

import pyarrow as pa

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.datatype import (
    DTypeMapping,
    dtype_to_plan,
    schema_with_dtypes,
)
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.shard import RowRangeDescriptor, Shard
from refiner.pipeline.sources.base import BaseSource, SourceUnit
from refiner.pipeline.sources.readers.utils import (
    DEFAULT_TARGET_SHARD_BYTES,
    PathSelection,
    decode_value,
    path_selection_map,
)
from refiner.utils import check_required_dependencies


class ZarrReader(BaseSource):
    """Read a Zarr group as one row, episode rows, or leading-axis rows."""

    name = "read_zarr"

    def __init__(
        self,
        input: DataFolderLike,
        *,
        arrays: PathSelection | None = None,
        attrs: PathSelection | None = None,
        row_ends: str | None = None,
        split_leading_axis: bool = False,
        leading_axis_row_size: int = 1,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
        index_column: str | None = "index",
        file_path_column: str | None = "file_path",
        dtypes: DTypeMapping | None = None,
    ):
        """Create a Zarr reader.

        Args:
            input: Zarr group path.
            arrays: Array selections as output-name to Zarr-path mapping, a
                single path, a path sequence, or None to discover all arrays.
            attrs: Attribute selections with the same shape as ``arrays``.
            row_ends: Optional Zarr array path containing cumulative end offsets.
                When set, emitted rows are whole source ranges and never split
                across these boundaries.
            split_leading_axis: Emit fixed-size leading-axis rows when no
                ``row_ends`` path is provided.
            leading_axis_row_size: Number of leading-axis items in each logical
                row when ``split_leading_axis`` is enabled.
            target_shard_bytes: Approximate byte target used to pack logical
                rows into shards in split modes.
            num_shards: Optional target shard count for split modes.
            index_column: Output metadata column containing the logical row
                index in split modes, or None to omit it.
            file_path_column: Output metadata column containing the source path,
                or None to omit it.
            dtypes: Optional dtype overrides for output columns.
        """
        self.root = DataFolder.resolve(input)
        check_required_dependencies("read_zarr", ["zarr"], dist="zarr")
        if row_ends is not None and split_leading_axis:
            raise ValueError("row_ends and split_leading_axis are mutually exclusive")
        if leading_axis_row_size <= 0:
            raise ValueError("leading_axis_row_size must be greater than zero")
        if leading_axis_row_size != 1 and not split_leading_axis:
            raise ValueError("leading_axis_row_size requires split_leading_axis=True")
        if target_shard_bytes <= 0:
            raise ValueError("target_shard_bytes must be greater than zero")
        if num_shards is not None and num_shards <= 0:
            raise ValueError("num_shards must be greater than zero")
        self.arrays = (
            None if arrays is None else path_selection_map(arrays, format_name="Zarr")
        )
        self.attrs = (
            None if attrs is None else path_selection_map(attrs, format_name="Zarr")
        )
        self.row_ends = row_ends
        self.split_leading_axis = split_leading_axis
        self.leading_axis_row_size = leading_axis_row_size
        self.target_shard_bytes = target_shard_bytes
        self.num_shards = num_shards
        self.index_column = index_column
        self.file_path_column = file_path_column
        self.dtypes = dtypes
        _validate_output_names(
            self.arrays or {},
            self.attrs or {},
            reserved=self._reserved_output_names(
                split=row_ends is not None or split_leading_axis
            ),
        )
        if (
            (row_ends is not None or split_leading_axis)
            and file_path_column is not None
            and file_path_column == index_column
        ):
            raise ValueError("file_path_column and index_column must be distinct")

    @property
    def schema(self) -> pa.Schema | None:
        return schema_with_dtypes(None, self.dtypes)

    def describe(self) -> dict[str, Any]:
        return {
            "path": self.root.abs_path(),
            "arrays": dict(self.arrays) if self.arrays is not None else None,
            "attrs": dict(self.attrs) if self.attrs is not None else None,
            "row_ends": self.row_ends,
            "split_leading_axis": self.split_leading_axis,
            "leading_axis_row_size": self.leading_axis_row_size,
            "target_shard_bytes": self.target_shard_bytes,
            "num_shards": self.num_shards,
            "index_column": self.index_column,
            "file_path_column": self.file_path_column,
            "dtypes": (
                {key: dtype_to_plan(dtype) for key, dtype in self.dtypes.items()}
                if self.dtypes
                else None
            ),
        }

    def list_shards(self) -> list[Shard]:
        path = self.root.abs_path()

        group = self._open_group()
        arrays = self._selected_arrays(group, validate_names=True)
        split_ranges = self._shard_ranges(group, arrays)
        return [
            Shard.from_row_range(
                start=start,
                end=end,
                global_ordinal=index,
                start_key=path,
                end_key=path,
            )
            for index, (start, end) in enumerate(split_ranges)
        ]

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        group = self._open_group()
        arrays = self._selected_arrays(group)
        if self.row_ends is not None:
            descriptor = shard.descriptor
            assert isinstance(descriptor, RowRangeDescriptor)
            source_ranges = self._row_end_ranges(
                group,
                arrays,
                row_start=descriptor.start,
                row_end=descriptor.end,
            )
            if not source_ranges:
                return
            block_start = source_ranges[0][0]
            block_end = source_ranges[-1][1]
            block = self._read_arrays(arrays, start=block_start, end=block_end)
            attrs = self._read_attrs(group)
            for row_index, (start, end) in zip(
                range(descriptor.start, descriptor.end),
                source_ranges,
                strict=True,
            ):
                row = self._row_metadata(index=row_index)
                row.update(
                    {
                        name: value[start - block_start : end - block_start]
                        for name, value in block.items()
                    }
                )
                row.update(attrs)
                yield DictRow(row)
            return

        if self.split_leading_axis:
            descriptor = shard.descriptor
            assert isinstance(descriptor, RowRangeDescriptor)
            raw_start = descriptor.start * self.leading_axis_row_size
            raw_end = descriptor.end * self.leading_axis_row_size
            block = self._read_arrays(
                arrays,
                start=raw_start,
                end=raw_end,
            )
            attrs = self._read_attrs(group)
            for row_index in range(descriptor.start, descriptor.end):
                offset = (row_index - descriptor.start) * self.leading_axis_row_size
                row = self._row_metadata(index=row_index)
                row.update(
                    {
                        name: value[offset : offset + self.leading_axis_row_size]
                        for name, value in block.items()
                    }
                )
                row.update(attrs)
                yield DictRow(row)
            return

        row = self._row_metadata(index=None)
        row.update(self._read_arrays(arrays))
        row.update(self._read_attrs(group))
        yield DictRow(row)

    def _open_group(self) -> Any:
        import zarr

        store = zarr.storage.FSStore(self.root._join(""), fs=self.root.fs, mode="r")
        return zarr.open_group(store=store, mode="r")

    def _reserved_output_names(self, *, split: bool) -> set[str]:
        names = set()
        if self.file_path_column is not None:
            names.add(self.file_path_column)
        if split and self.index_column is not None:
            names.add(self.index_column)
        return names

    def _row_metadata(self, *, index: int | None) -> dict[str, Any]:
        row: dict[str, Any] = {}
        if self.file_path_column is not None:
            row[self.file_path_column] = self.root.abs_path()
        if self.index_column is not None and index is not None:
            row[self.index_column] = index
        return row

    def _selected_arrays(
        self,
        group: Any,
        *,
        validate_names: bool = False,
    ) -> dict[str, Any]:
        paths = (
            self.arrays
            if self.arrays is not None
            else {
                path: path for path in _iter_array_paths(group) if path != self.row_ends
            }
        )
        if validate_names:
            _validate_output_names(
                paths,
                self.attrs or {},
                reserved=self._reserved_output_names(
                    split=self.row_ends is not None or self.split_leading_axis
                ),
            )
        arrays: dict[str, Any] = {}
        for output_name, path in paths.items():
            try:
                arrays[output_name] = group[path]
            except KeyError:
                raise KeyError(f"Zarr array not found: {path}") from None
        return arrays

    def _row_end_ranges(
        self,
        group: Any,
        arrays: Mapping[str, Any],
        *,
        row_start: int,
        row_end: int,
    ) -> list[tuple[int, int]]:
        if row_end < row_start:
            raise ValueError("Zarr shard row range is invalid")
        if row_start == row_end:
            return []
        row_ends_array = self._row_ends_array(group)
        read_start = max(0, row_start - 1)
        values = [int(value) for value in row_ends_array[read_start:row_end]]
        if len(values) != row_end - read_start:
            raise ValueError("Zarr shard row range exceeds row_ends length")
        ranges: list[tuple[int, int]] = []
        start = 0 if row_start == 0 else values[0]
        for end in values if row_start == 0 else values[1:]:
            if end < start:
                raise ValueError("Zarr row_ends must be monotonic increasing")
            ranges.append((start, end))
            start = end
        _check_final_end(arrays, ranges[-1][1], label="row_ends")
        return ranges

    def _shard_ranges(
        self,
        group: Any,
        arrays: Mapping[str, Any],
    ) -> list[tuple[int, int]]:
        if self.row_ends is None and not self.split_leading_axis:
            return [(0, 1)]

        if self.split_leading_axis:
            if not arrays:
                raise ValueError(
                    "split_leading_axis requires at least one selected array"
                )
            lengths: set[int] = set()
            for array in arrays.values():
                if not array.shape:
                    raise ValueError(
                        "Zarr selected arrays must have a leading dimension to split"
                    )
                lengths.add(int(array.shape[0]))
            if len(lengths) != 1:
                raise ValueError(
                    "Zarr selected arrays must have the same leading dimension"
                )
            length = lengths.pop()
            if length == 0:
                return []
            if length % self.leading_axis_row_size != 0:
                raise ValueError("Zarr leading dimension must be divisible by row size")
            row_count = length // self.leading_axis_row_size
            if self.num_shards is not None:
                step = ceil(row_count / self.num_shards)
            else:
                bytes_per_row = (
                    sum(_leading_item_bytes(array) for array in arrays.values())
                    * self.leading_axis_row_size
                )
                target_rows = max(1, self.target_shard_bytes // max(1, bytes_per_row))
                chunk_rows = max(
                    1,
                    ceil(
                        max(
                            int(array.chunks[0])
                            if array.chunks
                            else int(array.shape[0])
                            for array in arrays.values()
                        )
                        / self.leading_axis_row_size
                    ),
                )
                step = max(chunk_rows, (target_rows // chunk_rows) * chunk_rows)
            return [
                (start, min(start + step, row_count))
                for start in range(0, row_count, step)
            ]

        row_ends_array = self._row_ends_array(group)
        row_count = int(row_ends_array.shape[0])
        if row_count == 0:
            return []
        if self.num_shards is not None:
            final_end = _validate_row_ends(row_ends_array)
            _check_final_end(arrays, final_end, label="row_ends", exact=True)
            step = ceil(row_count / self.num_shards)
            return [
                (start, min(start + step, row_count))
                for start in range(0, row_count, step)
            ]

        bytes_per_step = sum(_leading_item_bytes(array) for array in arrays.values())
        if bytes_per_step <= 0:
            final_end = _validate_row_ends(row_ends_array)
            _check_final_end(arrays, final_end, label="row_ends", exact=True)
            return [(0, row_count)]

        ranges: list[tuple[int, int]] = []
        shard_start = 0
        current_bytes = 0
        previous_end = 0
        for row_index, end in _iter_row_ends(row_ends_array):
            if end < previous_end:
                raise ValueError("Zarr row_ends must be monotonic increasing")
            row_bytes = max(1, end - previous_end) * bytes_per_step
            if (
                row_index > shard_start
                and current_bytes + row_bytes > self.target_shard_bytes
            ):
                ranges.append((shard_start, row_index))
                shard_start = row_index
                current_bytes = 0
            current_bytes += row_bytes
            previous_end = end
        ranges.append((shard_start, row_count))
        _check_final_end(arrays, previous_end, label="row_ends", exact=True)
        return ranges

    def _row_ends_array(self, group: Any) -> Any:
        try:
            row_ends_array = group[self.row_ends]
        except KeyError:
            raise KeyError(f"Zarr row_ends array not found: {self.row_ends}") from None
        if len(row_ends_array.shape) != 1:
            raise ValueError("Zarr row_ends must be one-dimensional")
        return row_ends_array

    def _read_arrays(
        self,
        arrays: Mapping[str, Any],
        *,
        start: int | None = None,
        end: int | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for output_name, array in arrays.items():
            if start is not None:
                row[output_name] = array[start:end]
            elif array.shape == ():
                row[output_name] = array[()]
            else:
                row[output_name] = array[:]
        return row

    def _read_attrs(self, group: Any) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        for output_name, attr_name in (self.attrs or {}).items():
            if attr_name not in group.attrs:
                raise KeyError(f"Zarr attr not found: {attr_name}")
            attrs[output_name] = decode_value(group.attrs[attr_name])
        return attrs


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


def _iter_row_ends(array: Any) -> Iterator[tuple[int, int]]:
    chunk = max(1, int(array.chunks[0]) if array.chunks else 8192)
    length = int(array.shape[0])
    for start in range(0, length, chunk):
        for offset, value in enumerate(array[start : min(start + chunk, length)]):
            yield start + offset, int(value)


def _validate_row_ends(array: Any) -> int:
    previous_end = 0
    for _, end in _iter_row_ends(array):
        if end < previous_end:
            raise ValueError("Zarr row_ends must be monotonic increasing")
        previous_end = end
    return previous_end


def _check_final_end(
    arrays: Mapping[str, Any],
    final_end: int,
    *,
    label: str,
    exact: bool = False,
) -> None:
    for output_name, array in arrays.items():
        leading_length = int(array.shape[0])
        if final_end > leading_length:
            raise ValueError(
                f"Zarr {label} exceed leading dimension for {output_name!r}"
            )
        if exact and final_end != leading_length:
            raise ValueError(
                f"Zarr {label} end before leading dimension for {output_name!r}"
            )


def _leading_item_bytes(array: Any) -> int:
    trailing_shape = tuple(int(value) for value in array.shape[1:])
    return max(1, int(array.dtype.itemsize) * int(prod(trailing_shape or (1,))))


def _iter_array_paths(group: Any, prefix: str = "") -> Iterator[str]:
    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if hasattr(item, "shape"):
            yield path
        else:
            yield from _iter_array_paths(item, path)


__all__ = ["PathSelection", "ZarrReader"]
