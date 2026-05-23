from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pyarrow as pa

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.row import Row
from refiner.pipeline.sinks.base import BaseSink
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

    def write_shard_block(self, shard_id: str, block: Block) -> int:
        rows = block
        count = 0
        for row in rows:
            self._write_row(shard_id, row)
            count += 1
        return count

    def _write_row(self, shard_id: str, row: Row) -> None:
        arrays = self.arrays or _default_robotics_arrays(row)
        store = self._store(shard_id)
        lengths: list[int] = []
        for zarr_path, source_key in arrays.items():
            value = _row_value(row, source_key)
            if value is None:
                raise ValueError(f"Zarr source value is missing: {source_key}")
            array = _as_array(value)
            if array.ndim == 0:
                array = array.reshape(1)
            lengths.append(int(array.shape[0]))
            self._append_array(store, zarr_path, array)
        if lengths and self.episode_ends_path is not None:
            length = lengths[0]
            if any(item != length for item in lengths):
                raise ValueError("Zarr arrays for one row must have matching lengths")
            store.row_end += length
            self._append_array(
                store,
                self.episode_ends_path,
                np.asarray([store.row_end], dtype=np.int64),
            )

    def _store(self, shard_id: str) -> _ShardStore:
        relpath = self.store_template.format(
            shard_id=shard_id,
            worker_id=get_active_worker_token(),
        )
        store = self._stores.get(relpath)
        if store is not None:
            return store
        import zarr

        mode = "w" if self.overwrite else "w-"
        store = _ShardStore(zarr.open_group(self.output.abs_path(relpath), mode=mode))
        self._stores[relpath] = store
        return store

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
    if isinstance(value, pa.ChunkedArray | pa.Array):
        return np.asarray(value.to_pylist())
    if isinstance(value, Iterable) and not isinstance(value, str | bytes | np.ndarray):
        return np.asarray(list(cast(Iterable[Any], value)))
    return np.asarray(value)


__all__ = ["ZarrSink"]
