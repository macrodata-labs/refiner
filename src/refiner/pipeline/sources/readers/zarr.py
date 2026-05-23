from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import ceil, prod
from typing import Any

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

DEFAULT_TARGET_SHARD_BYTES = 256 * 1024**2


@dataclass(frozen=True, slots=True)
class _ArrayInfo:
    output_name: str
    path: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: Any

    @property
    def leading_length(self) -> int:
        if not self.shape:
            raise ValueError(
                f"Zarr array {self.path!r} must have a leading dimension to split"
            )
        return int(self.shape[0])

    @property
    def leading_chunk(self) -> int:
        return int(self.chunks[0]) if self.chunks else self.leading_length

    @property
    def bytes_per_step(self) -> int:
        trailing_shape = self.shape[1:]
        return max(1, int(self.dtype.itemsize) * int(prod(trailing_shape or (1,))))


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
    """Read a Zarr group as one row, episode rows, or leading-axis windows."""

    name = "read_zarr"

    def __init__(
        self,
        input: DataFolderLike,
        *,
        arrays: PathSelection | None = None,
        attrs: PathSelection | None = None,
        row_ends: str | None = None,
        split_leading_axis: bool = False,
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
            split_leading_axis: Emit aligned leading-axis windows when no
                ``row_ends`` path is provided.
            target_shard_bytes: Approximate byte target used to pack output rows
                into shards in split modes.
            num_shards: Optional target shard count for split modes.
            index_column: Output metadata column containing the episode/window
                index in split modes, or None to omit it.
            file_path_column: Output metadata column containing the source path,
                or None to omit it.
            dtypes: Optional dtype overrides for output columns.
        """
        self.root = DataFolder.resolve(input)
        check_required_dependencies("read_zarr", ["zarr"], dist="zarr")
        if row_ends is not None and split_leading_axis:
            raise ValueError("row_ends and split_leading_axis are mutually exclusive")
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
        self.target_shard_bytes = target_shard_bytes
        self.num_shards = num_shards
        self.index_column = index_column
        self.file_path_column = file_path_column
        self.dtypes = dtypes
        _validate_output_names(
            self.arrays or {},
            self.attrs or {},
            reserved=self._reserved_output_names(split=self._is_split_mode),
        )
        if (
            self._is_split_mode
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
        import zarr

        group = zarr.open_group(store=zarr_store(self.root), mode="r")
        arrays = self._array_selection(group)
        _validate_output_names(
            arrays,
            self.attrs or {},
            reserved=self._reserved_output_names(split=self._is_split_mode),
        )
        split_ranges = self._split_ranges(group, self._array_infos(group, arrays))
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
        import zarr

        group = zarr.open_group(store=zarr_store(self.root), mode="r")
        arrays = self._array_selection(group)
        _validate_output_names(
            arrays,
            self.attrs or {},
            reserved=self._reserved_output_names(split=self._is_split_mode),
        )
        if self._is_split_mode:
            descriptor = shard.descriptor
            assert isinstance(descriptor, RowRangeDescriptor)
            source_ranges = self._source_ranges(group, self._array_infos(group, arrays))
            for row_index in range(descriptor.start, descriptor.end):
                start, end = source_ranges[row_index]
                row = self._row_metadata(index=row_index)
                row.update(self._read_arrays(group, arrays, start=start, end=end))
                yield DictRow(self._read_attrs(group, row))
            return

        row = self._row_metadata(index=None)
        row.update(self._read_arrays(group, arrays))
        yield DictRow(self._read_attrs(group, row))

    @property
    def _is_split_mode(self) -> bool:
        return self.row_ends is not None or self.split_leading_axis

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

    def _array_selection(self, group: Any) -> dict[str, str]:
        if self.arrays is not None:
            return self.arrays
        return {
            path: path for path in _iter_array_paths(group) if path != self.row_ends
        }

    def _array_infos(
        self,
        group: Any,
        arrays: Mapping[str, str],
    ) -> list[_ArrayInfo]:
        infos: list[_ArrayInfo] = []
        for output_name, path in arrays.items():
            try:
                array = group[path]
            except KeyError:
                raise KeyError(f"Zarr array not found: {path}") from None
            infos.append(
                _ArrayInfo(
                    output_name=output_name,
                    path=path,
                    shape=tuple(int(value) for value in array.shape),
                    chunks=tuple(int(value) for value in array.chunks),
                    dtype=array.dtype,
                )
            )
        return infos

    def _source_ranges(
        self,
        group: Any,
        infos: Sequence[_ArrayInfo],
    ) -> list[tuple[int, int]]:
        if self.row_ends is not None:
            try:
                ends = [int(value) for value in group[self.row_ends][:]]
            except KeyError:
                raise KeyError(
                    f"Zarr row_ends array not found: {self.row_ends}"
                ) from None
            ranges = _ranges_from_ends(ends)
            _validate_source_ranges(ranges, infos, label="row_ends")
            return ranges
        return _leading_axis_ranges(
            infos,
            target_shard_bytes=self.target_shard_bytes,
            num_shards=self.num_shards,
        )

    def _split_ranges(
        self,
        group: Any,
        infos: Sequence[_ArrayInfo],
    ) -> list[tuple[int, int]]:
        if not self._is_split_mode:
            return [(0, 1)]
        if self.split_leading_axis:
            source_ranges = self._source_ranges(group, infos)
            return [(index, index + 1) for index in range(len(source_ranges))]
        return _pack_output_rows(
            self._source_ranges(group, infos),
            infos,
            target_shard_bytes=self.target_shard_bytes,
            num_shards=self.num_shards,
        )

    def _read_arrays(
        self,
        group: Any,
        arrays: Mapping[str, str],
        *,
        start: int | None = None,
        end: int | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for output_name, path in arrays.items():
            try:
                row[output_name] = (
                    group[path][start:end] if start is not None else group[path][:]
                )
            except KeyError:
                raise KeyError(f"Zarr array not found: {path}") from None
        return row

    def _read_attrs(self, group: Any, row: dict[str, Any]) -> dict[str, Any]:
        for output_name, attr_name in (self.attrs or {}).items():
            if attr_name not in group.attrs:
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


def _ranges_from_ends(ends: Sequence[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for end in ends:
        if end < start:
            raise ValueError("Zarr row_ends must be monotonic increasing")
        ranges.append((start, end))
        start = end
    return ranges


def _validate_source_ranges(
    ranges: Sequence[tuple[int, int]],
    infos: Sequence[_ArrayInfo],
    *,
    label: str,
) -> None:
    if not infos:
        return
    final_end = ranges[-1][1] if ranges else 0
    for info in infos:
        if final_end > info.leading_length:
            raise ValueError(
                f"Zarr {label} exceed leading dimension for {info.output_name!r}"
            )


def _leading_axis_ranges(
    infos: Sequence[_ArrayInfo],
    *,
    target_shard_bytes: int,
    num_shards: int | None,
) -> list[tuple[int, int]]:
    if not infos:
        raise ValueError("split_leading_axis requires at least one selected array")
    lengths = {info.leading_length for info in infos}
    if len(lengths) != 1:
        raise ValueError("Zarr selected arrays must have the same leading dimension")
    length = lengths.pop()
    if length == 0:
        return []
    if num_shards is not None:
        step = ceil(length / num_shards)
    else:
        bytes_per_step = sum(info.bytes_per_step for info in infos)
        target_steps = max(1, target_shard_bytes // max(1, bytes_per_step))
        heavy = max(infos, key=lambda info: info.bytes_per_step)
        base = max(1, heavy.leading_chunk)
        step = max(base, (target_steps // base) * base)
    return [(start, min(start + step, length)) for start in range(0, length, step)]


def _pack_output_rows(
    source_ranges: Sequence[tuple[int, int]],
    infos: Sequence[_ArrayInfo],
    *,
    target_shard_bytes: int,
    num_shards: int | None,
) -> list[tuple[int, int]]:
    if not source_ranges:
        return []
    if num_shards is not None:
        step = ceil(len(source_ranges) / num_shards)
        return [
            (start, min(start + step, len(source_ranges)))
            for start in range(0, len(source_ranges), step)
        ]

    bytes_per_step = sum(info.bytes_per_step for info in infos)
    if bytes_per_step <= 0:
        return [(0, len(source_ranges))]
    ranges: list[tuple[int, int]] = []
    start_index = 0
    current_bytes = 0
    for index, (start, end) in enumerate(source_ranges):
        row_bytes = max(1, end - start) * bytes_per_step
        if index > start_index and current_bytes + row_bytes > target_shard_bytes:
            ranges.append((start_index, index))
            start_index = index
            current_bytes = 0
        current_bytes += row_bytes
    ranges.append((start_index, len(source_ranges)))
    return ranges


def _iter_array_paths(group: Any, prefix: str = "") -> Iterator[str]:
    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if hasattr(item, "shape"):
            yield path
        else:
            yield from _iter_array_paths(item, path)


__all__ = ["PathSelection", "ZarrReader"]
