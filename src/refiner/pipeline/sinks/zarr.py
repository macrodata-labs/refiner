from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pyarrow as pa

from refiner.execution.asyncio.runtime import submit
from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.io.zarr import iter_zarr_array_paths, zarr_store
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.row import Row
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.reducer.file import FileCleanupReducerSink
from refiner.robotics.row import RoboticsRow
from refiner.utils import check_required_dependencies
from refiner.video import VideoSource
from refiner.worker.context import (
    get_active_stage_index,
    get_active_worker_token,
    get_finalized_workers,
)

_DEFAULT_ARRAY_CHUNK_LENGTH = 1024


@dataclass
class _ShardStore:
    root: Any
    arrays: dict[str, Any] = field(default_factory=dict)
    row_end: int = 0
    next_temp_index: int = 0


class ZarrSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        arrays: Mapping[str, str] | None = None,
        episode_ends_path: str | None = "meta/episode_ends",
        store_template: str = "{shard_id}__w{worker_id}.zarr",
        video_frame_batch_size: int = 8,
        reduce_to_single_store: bool = False,
        overwrite: bool = True,
    ):
        check_required_dependencies("write_zarr", ["zarr"], dist="zarr")
        if video_frame_batch_size <= 0:
            raise ValueError("video_frame_batch_size must be greater than zero")
        self.output = DataFolder.resolve(output)
        self.arrays = dict(arrays) if arrays is not None else None
        self.episode_ends_path = episode_ends_path
        if self.arrays is not None:
            _validate_array_paths(self.arrays, episode_ends_path)
        self.store_template = store_template
        self.video_frame_batch_size = video_frame_batch_size
        self.reduce_to_single_store = reduce_to_single_store
        self.overwrite = overwrite
        self._stores: dict[str, _ShardStore] = {}
        self._default_arrays: dict[str, str] | None = None

    def write_shard_block(self, shard_id: str, block: Block) -> int:
        count = 0
        for row in block:
            self._write_row(shard_id, row)
            count += 1
        return count

    def _write_row(self, shard_id: str, row: Row) -> None:
        arrays = self._arrays_for_row(row)
        row_arrays: dict[str, np.ndarray] = {}
        row_videos: list[tuple[str, VideoSource]] = []
        lengths: list[int] = []
        store: _ShardStore | None = None
        for zarr_path, source_key in arrays.items():
            value = _row_value(row, source_key)
            if value is None:
                raise ValueError(f"Zarr source value is missing: {source_key}")
            if store is None:
                store = self._store(shard_id)
            if isinstance(value, VideoSource):
                row_videos.append((zarr_path, value))
                continue
            array = _as_array(value)
            if array.ndim == 0:
                array = array.reshape(1)
            lengths.append(int(array.shape[0]))
            row_arrays[zarr_path] = array
        if not lengths:
            expected_length = None
        else:
            expected_length = lengths[0]
            if any(item != expected_length for item in lengths):
                raise ValueError("Zarr arrays for one row must have matching lengths")
        temp_videos: list[tuple[str, str, int]] = []
        if store is not None:
            try:
                for zarr_path, video in row_videos:
                    temp_path = self._temp_path(store, zarr_path)
                    temp_videos.append((zarr_path, temp_path, 0))
                    video_length = submit(
                        self._append_video(
                            store,
                            temp_path,
                            video,
                            expected_length=expected_length,
                        )
                    ).result()
                    lengths.append(video_length)
                    temp_videos[-1] = (zarr_path, temp_path, video_length)
                if lengths:
                    length = lengths[0]
                    if any(item != length for item in lengths):
                        raise ValueError(
                            "Zarr arrays for one row must have matching lengths"
                        )
                for zarr_path, array in row_arrays.items():
                    self._validate_array_append(store, zarr_path, array)
                for zarr_path, temp_path, _ in temp_videos:
                    self._validate_array_append(
                        store, zarr_path, store.arrays[temp_path]
                    )
                for zarr_path, array in row_arrays.items():
                    self._append_array(store, zarr_path, array)
                for zarr_path, temp_path, _ in temp_videos:
                    self._copy_temp_array(store, temp_path, zarr_path)
                if lengths and self.episode_ends_path is not None:
                    store.row_end += lengths[0]
                    self._append_array(
                        store,
                        self.episode_ends_path,
                        np.asarray([store.row_end], dtype=np.int64),
                    )
            finally:
                for _, temp_path, _ in temp_videos:
                    self._drop_array(store, temp_path)
        return

    async def _append_video(
        self,
        store: _ShardStore,
        path: str,
        video: VideoSource,
        *,
        expected_length: int | None = None,
    ) -> int:
        batch: list[np.ndarray] = []
        count = 0
        async for frame in _iter_video_frame_arrays(video):
            batch.append(np.asarray(frame))
            if len(batch) >= self.video_frame_batch_size:
                self._append_array(store, path, np.stack(batch, axis=0))
                count += len(batch)
                batch.clear()
                if expected_length is not None and count > expected_length:
                    raise ValueError(
                        "Zarr arrays for one row must have matching lengths"
                    )
        if batch:
            self._append_array(store, path, np.stack(batch, axis=0))
            count += len(batch)
        if expected_length is not None and count != expected_length:
            raise ValueError("Zarr arrays for one row must have matching lengths")
        return count

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
        relpath = self._store_relpath(shard_id)
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

    def _store_relpath(self, shard_id: str) -> str:
        relpath = self.store_template.format(
            shard_id=shard_id,
            worker_id=get_active_worker_token(),
        )
        return f"_parts/{relpath}" if self.reduce_to_single_store else relpath

    def on_shard_complete(self, shard_id: str) -> None:
        self._stores.pop(self._store_relpath(shard_id), None)

    def _temp_path(self, store: _ShardStore, path: str) -> str:
        temp_path = f"__tmp/{store.next_temp_index}/{path}"
        store.next_temp_index += 1
        return temp_path

    def _append_array(
        self,
        store: _ShardStore,
        path: str,
        array: np.ndarray,
    ) -> None:
        dataset = store.arrays.get(path)
        if dataset is None:
            chunks = (_DEFAULT_ARRAY_CHUNK_LENGTH, *array.shape[1:])
            dataset = store.root.create_dataset(
                path,
                shape=(0, *array.shape[1:]),
                chunks=chunks,
                dtype=array.dtype,
            )
            store.arrays[path] = dataset
        else:
            self._validate_array_append(store, path, array)
        dataset.append(array, axis=0)

    def _validate_array_append(
        self,
        store: _ShardStore,
        path: str,
        array: np.ndarray,
    ) -> None:
        dataset = store.arrays.get(path)
        if dataset is None:
            return
        if tuple(dataset.shape[1:]) != tuple(array.shape[1:]):
            raise ValueError(
                f"Zarr arrays for {path!r} must have matching trailing shapes"
            )
        if dataset.dtype != array.dtype:
            raise ValueError(f"Zarr arrays for {path!r} must have matching dtypes")

    def _copy_temp_array(self, store: _ShardStore, temp_path: str, path: str) -> None:
        source = store.arrays[temp_path]
        for start in range(0, int(source.shape[0]), _merge_batch_size(source)):
            end = min(int(source.shape[0]), start + _merge_batch_size(source))
            self._append_array(store, path, np.asarray(source[start:end]))

    def _drop_array(self, store: _ShardStore, path: str) -> None:
        store.arrays.pop(path, None)
        if path.startswith("__tmp/"):
            path = "/".join(path.split("/")[:2])
            for key in list(store.arrays):
                if key == path or key.startswith(f"{path}/"):
                    store.arrays.pop(key, None)
        try:
            del store.root[path]
        except (KeyError, FileNotFoundError):
            pass
        if path.startswith("__tmp/"):
            try:
                del store.root["__tmp"]
            except (KeyError, FileNotFoundError):
                pass

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
                "video_frame_batch_size": self.video_frame_batch_size,
                "reduce_to_single_store": self.reduce_to_single_store,
                "overwrite": self.overwrite,
            },
        )

    def build_reducer(self) -> BaseSink | None:
        if self.reduce_to_single_store:
            return ZarrMergeReducerSink(
                output=self.output,
                store_template=self.store_template,
                episode_ends_path=self.episode_ends_path,
                overwrite=self.overwrite,
            )
        return FileCleanupReducerSink(
            output=self.output,
            filename_template=self.store_template,
            reducer_name="write_zarr_reduce",
            recursive=True,
        )


class ZarrMergeReducerSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        store_template: str,
        episode_ends_path: str | None,
        overwrite: bool,
    ) -> None:
        check_required_dependencies("write_zarr", ["zarr"], dist="zarr")
        self.output = DataFolder.resolve(output)
        self.store_template = store_template
        self.episode_ends_path = episode_ends_path
        self.overwrite = overwrite
        self._merged = False

    @property
    def counts_output_rows(self) -> bool:
        return False

    def write_shard_block(self, shard_id, block) -> None:
        del shard_id, block
        self._merge()

    def describe(self) -> tuple[str, str, dict[str, object]]:
        return (
            "write_zarr_reduce",
            "writer",
            {
                "path": self.output.abs_path(),
                "store_template": self.store_template,
                "reduce_to_single_store": True,
            },
        )

    def _merge(self) -> None:
        if self._merged:
            return
        self._merged = True

        stage_index = get_active_stage_index()
        if stage_index is None or stage_index <= 0:
            raise ValueError(
                "write_zarr_reduce requires an active reducer stage with a prior writer stage"
            )

        import zarr

        final = zarr.open_group(
            store=zarr_store(self.output, "", mode="a"),
            mode="a",
        )
        if self.overwrite:
            _clear_final_group(final)

        stores = sorted(
            get_finalized_workers(stage_index=stage_index - 1),
            key=lambda row: (
                row.global_ordinal is None,
                row.global_ordinal if row.global_ordinal is not None else row.shard_id,
            ),
        )
        row_offset = 0
        arrays: dict[str, Any] = {}
        for row in stores:
            relpath = self._part_relpath(row.shard_id, row.worker_token)
            source = zarr.open_group(
                store=zarr_store(self.output, relpath, mode="r"),
                mode="r",
            )
            for path in iter_zarr_array_paths(source):
                source_array = source[path]
                if path == self.episode_ends_path:
                    if source_array.shape[0] == 0:
                        continue
                    values = np.asarray(source_array[:], dtype=np.int64)
                    _append_reduced_array(
                        final,
                        arrays,
                        path,
                        values + row_offset,
                        source_array,
                    )
                    row_offset += int(values[-1])
                    continue
                for start in range(
                    0, int(source_array.shape[0]), _merge_batch_size(source_array)
                ):
                    end = min(
                        int(source_array.shape[0]),
                        start + _merge_batch_size(source_array),
                    )
                    _append_reduced_array(
                        final,
                        arrays,
                        path,
                        np.asarray(source_array[start:end]),
                        source_array,
                    )

        try:
            self.output.rm("_parts", recursive=True)
        except FileNotFoundError:
            pass

    def _part_relpath(self, shard_id: str, worker_token: str) -> str:
        return "_parts/" + self.store_template.format(
            shard_id=shard_id,
            worker_id=worker_token,
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
                if video is None:
                    raise
                return video
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


async def _iter_video_frame_arrays(video: VideoSource):
    iter_frame_arrays = getattr(video, "iter_frame_arrays", None)
    if callable(iter_frame_arrays):
        frames = iter_frame_arrays()
        if hasattr(frames, "__aiter__"):
            async for frame in frames:
                yield frame
            return
        for frame in frames:
            yield frame
        return
    async for frame in video.iter_frames():
        yield frame.frame.to_ndarray(format="rgb24")


def _clear_final_group(group: Any) -> None:
    for key in sorted({*group.array_keys(), *group.group_keys()}):
        if key != "_parts":
            del group[key]


def _merge_batch_size(array: Any) -> int:
    chunks = getattr(array, "chunks", None)
    if isinstance(chunks, tuple) and chunks and isinstance(chunks[0], int):
        return max(1, int(chunks[0]))
    return max(1, min(int(array.shape[0]), 1024))


def _append_reduced_array(
    root: Any,
    arrays: dict[str, Any],
    path: str,
    values: np.ndarray,
    source_array: Any,
) -> None:
    dataset = arrays.get(path)
    if dataset is None:
        dataset = root.create_dataset(
            path,
            shape=(0, *values.shape[1:]),
            chunks=getattr(source_array, "chunks", None),
            dtype=source_array.dtype,
            compressor=getattr(source_array, "compressor", None),
        )
        arrays[path] = dataset
    dataset.append(values, axis=0)


__all__ = ["ZarrMergeReducerSink", "ZarrSink"]
