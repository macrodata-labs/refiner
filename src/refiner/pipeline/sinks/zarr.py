from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pyarrow as pa

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.io.zarr import zarr_store
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.row import Row
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.reducer.file import FileCleanupReducerSink
from refiner.robotics.row import RoboticsRow
from refiner.utils import check_required_dependencies
from refiner.worker.context import get_active_worker_token


@dataclass
class _ShardStore:
    root: Any
    arrays: dict[str, Any] = field(default_factory=dict)
    row_end: int = 0


class ZarrSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        arrays: Mapping[str, str] | None = None,
        episode_ends_path: str | None = "meta/episode_ends",
        store_template: str = "{shard_id}__w{worker_id}.zarr",
        overwrite: bool = True,
    ):
        check_required_dependencies("write_zarr", ["zarr"], dist="zarr")
        self.output = DataFolder.resolve(output)
        self.arrays = dict(arrays) if arrays is not None else None
        self.episode_ends_path = episode_ends_path
        self.store_template = store_template
        self.overwrite = overwrite
        self._stores: dict[str, _ShardStore] = {}
        self._default_arrays: dict[str, str] | None = None

    def write_shard_block(self, shard_id: str, block: Block) -> int:
        arrays_by_path: dict[str, list[np.ndarray]] = {}
        episode_lengths: list[int] = []
        count = 0
        for row in block:
            length = self._collect_row(row, arrays_by_path)
            if length is not None:
                episode_lengths.append(length)
            count += 1
        if arrays_by_path:
            store = self._store(shard_id)
            for zarr_path, arrays in arrays_by_path.items():
                self._append_array(store, zarr_path, np.concatenate(arrays, axis=0))
            if self.episode_ends_path is not None and episode_lengths:
                episode_ends = store.row_end + np.cumsum(
                    np.asarray(episode_lengths, dtype=np.int64)
                )
                store.row_end = int(episode_ends[-1])
                self._append_array(store, self.episode_ends_path, episode_ends)
        return count

    def _collect_row(
        self,
        row: Row,
        arrays_by_path: dict[str, list[np.ndarray]],
    ) -> int | None:
        arrays = self._arrays_for_row(row)
        lengths: list[int] = []
        for zarr_path, source_key in arrays.items():
            value = _row_value(row, source_key)
            if value is None:
                raise ValueError(f"Zarr source value is missing: {source_key}")
            array = _as_array(value)
            if array.ndim == 0:
                array = array.reshape(1)
            lengths.append(int(array.shape[0]))
            arrays_by_path.setdefault(zarr_path, []).append(array)
        if not lengths:
            return None
        length = lengths[0]
        if any(item != length for item in lengths):
            raise ValueError("Zarr arrays for one row must have matching lengths")
        return length

    def _arrays_for_row(self, row: Row) -> dict[str, str]:
        if self.arrays is not None:
            return self.arrays
        if self._default_arrays is None:
            self._default_arrays = _default_robotics_arrays(row)
        return self._default_arrays

    def _store(self, shard_id: str) -> _ShardStore:
        relpath = self.store_template.format(
            shard_id=shard_id,
            worker_id=get_active_worker_token(),
        )
        store = self._stores.get(relpath)
        if store is not None:
            return store
        mode = "w" if self.overwrite else "w-"
        import zarr

        store = _ShardStore(
            zarr.open_group(
                store=zarr_store(self.output, relpath, mode=mode), mode=mode
            )
        )
        self._stores[relpath] = store
        return store

    def on_shard_complete(self, shard_id: str) -> None:
        relpath = self.store_template.format(
            shard_id=shard_id,
            worker_id=get_active_worker_token(),
        )
        self._stores.pop(relpath, None)

    def _append_array(
        self,
        store: _ShardStore,
        path: str,
        array: np.ndarray,
    ) -> None:
        dataset = store.arrays.get(path)
        if dataset is None:
            chunks = (max(1, min(int(array.shape[0]), 1024)), *array.shape[1:])
            dataset = store.root.create_dataset(
                path,
                shape=(0, *array.shape[1:]),
                chunks=chunks,
                dtype=array.dtype,
            )
            store.arrays[path] = dataset
        dataset.append(array, axis=0)

    def close(self) -> None:
        self._stores.clear()

    def describe(self) -> tuple[str, str, dict[str, object]]:
        return (
            "write_zarr",
            "writer",
            {
                "path": self.output.abs_path(),
                "arrays": dict(self.arrays) if self.arrays is not None else None,
                "episode_ends_path": self.episode_ends_path,
                "store_template": self.store_template,
                "overwrite": self.overwrite,
            },
        )

    def build_reducer(self) -> BaseSink | None:
        return FileCleanupReducerSink(
            output=self.output,
            filename_template=self.store_template,
            reducer_name="write_zarr_reduce",
            recursive=True,
        )


def _default_robotics_arrays(row: Row) -> dict[str, str]:
    if not isinstance(row, RoboticsRow):
        raise ValueError("write_zarr requires arrays=... for non-RoboticsRow inputs")
    arrays: dict[str, str] = {}
    if row.actions is not None:
        arrays["data/action"] = "action"
    if row.states is not None:
        arrays["data/observation.state"] = "observation.state"
    if row.timestamps is not None:
        arrays["data/timestamp"] = "timestamp"
    return arrays


def _row_value(row: Row, key: str) -> Any:
    if isinstance(row, RoboticsRow):
        if key == "action":
            return row.actions
        if key == "observation.state":
            return row.states
        if key == "timestamp":
            return row.timestamps
        if key.startswith("observation."):
            return row.observations(key)
    return row[key]


def _as_array(value: Any) -> np.ndarray:
    if isinstance(value, pa.ChunkedArray):
        return _as_array(value.combine_chunks())
    if isinstance(value, pa.Array):
        if pa.types.is_primitive(value.type):
            return value.to_numpy(zero_copy_only=False)
        return np.asarray(value.to_pylist())
    if isinstance(value, Iterable) and not isinstance(value, str | bytes | np.ndarray):
        return np.asarray(list(cast(Iterable[Any], value)))
    return np.asarray(value)


__all__ = ["ZarrSink"]
