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
from refiner.video import VideoFrameArray
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
        if self.arrays is not None:
            _validate_array_paths(self.arrays, episode_ends_path)
        self.store_template = store_template
        self.overwrite = overwrite
        self._stores: dict[str, _ShardStore] = {}
        self._default_arrays: dict[str, str] | None = None

    def write_shard_block(self, shard_id: str, block: Block) -> int:
        count = 0
        for row in block:
            row_arrays, length = self._row_arrays(row)
            if row_arrays:
                store = self._store(shard_id)
                self._validate_store_append(store, row_arrays)
                for zarr_path, array in row_arrays.items():
                    self._append_array(store, zarr_path, array)
                if self.episode_ends_path is not None and length is not None:
                    store.row_end += length
                    self._append_array(
                        store,
                        self.episode_ends_path,
                        np.asarray([store.row_end], dtype=np.int64),
                    )
            count += 1
        return count

    def _row_arrays(self, row: Row) -> tuple[dict[str, np.ndarray], int | None]:
        arrays = self._arrays_for_row(row)
        row_arrays: dict[str, np.ndarray] = {}
        lengths: list[int] = []
        for zarr_path, source_key in arrays.items():
            value = _row_value(row, source_key)
            if value is None:
                raise ValueError(f"Zarr source value is missing: {source_key}")
            array = _as_array(value)
            if array.ndim == 0:
                array = array.reshape(1)
            lengths.append(int(array.shape[0]))
            row_arrays[zarr_path] = array
        if not lengths:
            return {}, None
        length = lengths[0]
        if any(item != length for item in lengths):
            raise ValueError("Zarr arrays for one row must have matching lengths")
        return row_arrays, length

    def _arrays_for_row(self, row: Row) -> dict[str, str]:
        if self.arrays is not None:
            return self.arrays
        default_arrays = _default_robotics_arrays(row)
        if self._default_arrays is None:
            self._default_arrays = default_arrays
            _validate_array_paths(self._default_arrays, self.episode_ends_path)
        elif default_arrays != self._default_arrays:
            raise ValueError(
                "Zarr default arrays changed across rows; pass arrays=... "
                "to write an explicit stable schema"
            )
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

    def _validate_store_append(
        self,
        store: _ShardStore,
        row_arrays: Mapping[str, np.ndarray],
    ) -> None:
        for path, array in row_arrays.items():
            dataset = store.arrays.get(path)
            if dataset is None:
                continue
            if tuple(dataset.shape[1:]) != tuple(array.shape[1:]):
                raise ValueError(
                    f"Zarr arrays for {path!r} must have matching trailing shapes"
                )
            if dataset.dtype != array.dtype:
                raise ValueError(f"Zarr arrays for {path!r} must have matching dtypes")

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


def _validate_array_paths(
    arrays: Mapping[str, str],
    episode_ends_path: str | None,
) -> None:
    if episode_ends_path is not None and episode_ends_path in arrays:
        raise ValueError(
            f"Zarr array path collides with episode_ends_path: {episode_ends_path}"
        )


def _row_value(row: Row, key: str) -> Any:
    if isinstance(row, RoboticsRow):
        if key == "action":
            return row.actions
        if key == "observation.state":
            return row.states
        if key == "timestamp":
            return row.timestamps
        if key.startswith("observation."):
            try:
                return row.observations(key)
            except KeyError:
                video = row.videos.get(key)
                if isinstance(video, VideoFrameArray):
                    return np.asarray(list(video.iter_frame_arrays()))
                if video is not None:
                    raise ValueError(
                        "write_zarr can only materialize video observations backed "
                        f"by frame arrays, got {type(video).__name__} for {key!r}"
                    )
                raise
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
