from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from string import Formatter
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
from refiner.video import VideoFrameArray, VideoSource
from refiner.worker.context import (
    get_active_job_id,
    get_active_stage_index,
    get_active_worker_token,
    get_finalized_workers,
)

_DEFAULT_ARRAY_CHUNK_BYTES = 8 * 1024 * 1024


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
        video_frame_batch_size: int = 8,
        array_chunk_bytes: int = _DEFAULT_ARRAY_CHUNK_BYTES,
        reduce_array_batch_bytes: int | None = None,
        reduce_to_single_store: bool = False,
        overwrite: bool = True,
    ):
        check_required_dependencies("write_zarr", ["zarr"], dist="zarr")
        if video_frame_batch_size <= 0:
            raise ValueError("video_frame_batch_size must be greater than zero")
        if array_chunk_bytes <= 0:
            raise ValueError("array_chunk_bytes must be greater than zero")
        if reduce_array_batch_bytes is not None and reduce_array_batch_bytes <= 0:
            raise ValueError("reduce_array_batch_bytes must be greater than zero")
        _validate_store_template(store_template)
        self.output = DataFolder.resolve(output)
        self.arrays = dict(arrays) if arrays is not None else None
        self.episode_ends_path = episode_ends_path
        if self.arrays is not None:
            if not self.arrays:
                raise ValueError("write_zarr arrays must not be empty")
            _validate_array_paths(self.arrays, episode_ends_path)
        self.store_template = store_template
        self.video_frame_batch_size = video_frame_batch_size
        self.array_chunk_bytes = array_chunk_bytes
        self.reduce_array_batch_bytes = (
            array_chunk_bytes
            if reduce_array_batch_bytes is None
            else reduce_array_batch_bytes
        )
        self.reduce_to_single_store = reduce_to_single_store
        self.overwrite = overwrite
        self._stores: dict[str, _ShardStore] = {}
        self._default_arrays: dict[str, str] | None = None
        self._checked_no_overwrite = False

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
        if store is not None:
            rollback_lengths: dict[str, int | None] = {}
            for zarr_path in [*row_arrays, *(path for path, _ in row_videos)]:
                dataset = store.arrays.get(zarr_path)
                rollback_lengths[zarr_path] = (
                    None if dataset is None else int(dataset.shape[0])
                )
            try:
                for zarr_path, array in row_arrays.items():
                    self._validate_array_append(store, zarr_path, array)
                for zarr_path, array in row_arrays.items():
                    self._append_array(store, zarr_path, array)
                for zarr_path, video in row_videos:
                    video_length = submit(
                        self._append_video(
                            store,
                            zarr_path,
                            video,
                            expected_length=expected_length,
                        )
                    ).result()
                    lengths.append(video_length)
                if lengths:
                    length = lengths[0]
                    if any(item != length for item in lengths):
                        raise ValueError(
                            "Zarr arrays for one row must have matching lengths"
                        )
                if lengths and self.episode_ends_path is not None:
                    dataset = store.arrays.get(self.episode_ends_path)
                    rollback_lengths[self.episode_ends_path] = (
                        None if dataset is None else int(dataset.shape[0])
                    )
                    store.row_end += lengths[0]
                    self._append_array(
                        store,
                        self.episode_ends_path,
                        np.asarray([store.row_end], dtype=np.int64),
                    )
            except Exception:
                for zarr_path, length in rollback_lengths.items():
                    if length is None:
                        self._drop_array(store, zarr_path)
                        continue
                    dataset = store.arrays.get(zarr_path)
                    if dataset is not None:
                        dataset.resize((length, *dataset.shape[1:]))
                if self.episode_ends_path is not None:
                    dataset = store.arrays.get(self.episode_ends_path)
                    store.row_end = (
                        0
                        if dataset is None or dataset.shape[0] == 0
                        else int(dataset[-1])
                    )
                raise
        return

    async def _append_video(
        self,
        store: _ShardStore,
        path: str,
        video: VideoSource,
        *,
        expected_length: int | None = None,
    ) -> int:
        if isinstance(video, VideoFrameArray):
            frames = video.frame_arrays
            if expected_length is not None and int(frames.shape[0]) != expected_length:
                raise ValueError("Zarr arrays for one row must have matching lengths")
            for start in range(0, int(frames.shape[0]), self.video_frame_batch_size):
                end = min(int(frames.shape[0]), start + self.video_frame_batch_size)
                self._append_array(store, path, frames[start:end])
            return int(frames.shape[0])

        batch: list[np.ndarray] = []
        count = 0
        async for frame in video.iter_frames():
            batch.append(frame.frame.to_ndarray(format="rgb24"))
            if len(batch) >= self.video_frame_batch_size:
                if expected_length is not None and count + len(batch) > expected_length:
                    raise ValueError(
                        "Zarr arrays for one row must have matching lengths"
                    )
                self._append_array(store, path, np.stack(batch, axis=0))
                count += len(batch)
                batch.clear()
        if batch:
            if expected_length is not None and count + len(batch) > expected_length:
                raise ValueError("Zarr arrays for one row must have matching lengths")
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
            if not self._default_arrays:
                raise ValueError(
                    "write_zarr inferred no default robotics arrays; pass arrays=..."
                )
            _validate_array_paths(self._default_arrays, self.episode_ends_path)
        elif default_arrays != self._default_arrays:
            raise ValueError(
                "Zarr default arrays changed across rows; pass arrays=... "
                "to write an explicit stable schema"
            )
        return self._default_arrays

    def _store(self, shard_id: str) -> _ShardStore:
        self._check_no_overwrite_output()
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

    def _check_no_overwrite_output(self) -> None:
        if self.overwrite or self._checked_no_overwrite:
            return
        self._checked_no_overwrite = True
        try:
            entries = self.output.ls("", detail=False)
        except FileNotFoundError:
            return
        if entries:
            raise ValueError("write_zarr output already exists and overwrite=False")

    def _store_relpath(self, shard_id: str) -> str:
        relpath = self.store_template.format(
            shard_id=shard_id,
            worker_id=get_active_worker_token(),
        )
        if self.reduce_to_single_store:
            return _part_store_relpath(relpath)
        return relpath

    def on_shard_complete(self, shard_id: str) -> None:
        self._stores.pop(self._store_relpath(shard_id), None)

    def _append_array(
        self,
        store: _ShardStore,
        path: str,
        array: np.ndarray,
    ) -> None:
        dataset = store.arrays.get(path)
        if dataset is None:
            dataset = store.root.create_dataset(
                path,
                shape=(0, *array.shape[1:]),
                chunks=_chunk_shape(array, self.array_chunk_bytes),
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

    def _drop_array(self, store: _ShardStore, path: str) -> None:
        store.arrays.pop(path, None)
        try:
            del store.root[path]
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
                "array_chunk_bytes": self.array_chunk_bytes,
                "reduce_array_batch_bytes": self.reduce_array_batch_bytes,
                "reduce_to_single_store": self.reduce_to_single_store,
                "overwrite": self.overwrite,
            },
        )

    def build_reducer(self) -> BaseSink | None:
        if self.reduce_to_single_store:
            return _ZarrMergeReducerSink(
                output=self.output,
                store_template=self.store_template,
                episode_ends_path=self.episode_ends_path,
                reduce_array_batch_bytes=self.reduce_array_batch_bytes,
                overwrite=self.overwrite,
            )
        return FileCleanupReducerSink(
            output=self.output,
            filename_template=self.store_template,
            reducer_name="write_zarr_reduce",
            recursive=True,
        )


class _ZarrMergeReducerSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        store_template: str,
        episode_ends_path: str | None,
        reduce_array_batch_bytes: int,
        overwrite: bool,
    ) -> None:
        check_required_dependencies("write_zarr", ["zarr"], dist="zarr")
        self.output = DataFolder.resolve(output)
        self.store_template = store_template
        self.episode_ends_path = episode_ends_path
        self.reduce_array_batch_bytes = reduce_array_batch_bytes
        self.overwrite = overwrite
        self._merged = False

    @property
    def counts_output_rows(self) -> bool:
        return False

    def write_shard_block(self, shard_id, block) -> None:
        del shard_id, block
        try:
            self._merge()
        except Exception:
            self._remove_current_parts()
            raise

    def describe(self) -> tuple[str, str, dict[str, object]]:
        return (
            "write_zarr_reduce",
            "writer",
            {
                "path": self.output.abs_path(),
                "store_template": self.store_template,
                "reduce_array_batch_bytes": self.reduce_array_batch_bytes,
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
        elif _group_has_payload(final):
            raise ValueError("write_zarr output already exists and overwrite=False")

        stores = sorted(
            get_finalized_workers(stage_index=stage_index - 1),
            key=lambda row: (
                row.global_ordinal is None,
                row.global_ordinal if row.global_ordinal is not None else row.shard_id,
            ),
        )
        row_offset = 0
        arrays: dict[str, Any] = {}
        payload_paths: set[str] | None = None
        for row in stores:
            relpath = self._part_relpath(row.shard_id, row.worker_token)
            if not self.output.exists(relpath):
                continue
            source = zarr.open_group(
                store=zarr_store(self.output, relpath, mode="r"),
                mode="r",
            )
            source_paths = set(iter_zarr_array_paths(source))
            source_payload_paths = {
                path for path in source_paths if path != self.episode_ends_path
            }
            if payload_paths is None:
                payload_paths = source_payload_paths
            elif source_payload_paths != payload_paths:
                raise ValueError(
                    "Zarr part stores must contain the same payload arrays"
                )
            for path in sorted(source_paths):
                source_array = source[path]
                if path == self.episode_ends_path:
                    if source_array.shape[0] == 0:
                        continue
                    part_last = row_offset
                    batch_size = _batch_length(
                        source_array,
                        self.reduce_array_batch_bytes,
                    )
                    for start in range(0, int(source_array.shape[0]), batch_size):
                        end = min(int(source_array.shape[0]), start + batch_size)
                        values = np.asarray(source_array[start:end], dtype=np.int64)
                        _append_reduced_array(
                            final,
                            arrays,
                            path,
                            values + row_offset,
                            source_array,
                        )
                        part_last = int(values[-1])
                    row_offset += part_last
                    continue
                batch_size = _batch_length(source_array, self.reduce_array_batch_bytes)
                for start in range(0, int(source_array.shape[0]), batch_size):
                    end = min(int(source_array.shape[0]), start + batch_size)
                    _append_reduced_array(
                        final,
                        arrays,
                        path,
                        np.asarray(source_array[start:end]),
                        source_array,
                    )

        self._remove_current_parts()
        if self.overwrite:
            self._remove_stale_parts()
        try:
            if not self.output.ls("_parts"):
                self.output.rmdir("_parts")
        except (FileNotFoundError, OSError, ValueError):
            pass

    def _part_relpath(self, shard_id: str, worker_token: str) -> str:
        return _part_store_relpath(
            self.store_template.format(
                shard_id=shard_id,
                worker_id=worker_token,
            )
        )

    def _remove_current_parts(self) -> None:
        try:
            self.output.rm(f"_parts/{get_active_job_id()}", recursive=True)
        except FileNotFoundError:
            pass

    def _remove_stale_parts(self) -> None:
        try:
            for path in self.output.ls("_parts", detail=False):
                if path != f"_parts/{get_active_job_id()}":
                    self.output.rm(path, recursive=True)
        except FileNotFoundError:
            pass


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


def _validate_store_template(store_template: str) -> None:
    fields = {
        field_name
        for _literal_text, field_name, _format_spec, _conversion in Formatter().parse(
            store_template
        )
        if field_name is not None
    }
    missing_fields = {"shard_id", "worker_id"}.difference(fields)
    if missing_fields:
        raise ValueError(
            "store_template requires fields: "
            + ", ".join(f"{{{field_name}}}" for field_name in sorted(missing_fields))
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


def _part_store_relpath(relpath: str) -> str:
    return f"_parts/{get_active_job_id()}/{relpath}"


def _clear_final_group(group: Any) -> None:
    for key in sorted({*group.array_keys(), *group.group_keys()}):
        if key != "_parts":
            del group[key]


def _group_has_payload(group: Any) -> bool:
    return any(key != "_parts" for key in {*group.array_keys(), *group.group_keys()})


def _chunk_shape(array: np.ndarray, target_bytes: int) -> tuple[int, ...]:
    return (_batch_length(array, target_bytes), *array.shape[1:])


def _batch_length(array: Any, target_bytes: int) -> int:
    dtype = np.dtype(array.dtype)
    row_values = int(np.prod(tuple(array.shape[1:]), dtype=np.int64))
    row_bytes = max(1, dtype.itemsize * max(1, row_values))
    return max(1, target_bytes // row_bytes)


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


__all__ = ["ZarrSink"]
