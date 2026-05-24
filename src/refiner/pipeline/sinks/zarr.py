from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from string import Formatter
from typing import Any, cast

import numpy as np
import pyarrow as pa

from refiner.execution.asyncio.runtime import submit
from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.row import Row
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.reducer.file import FileCleanupReducerSink
from refiner.robotics.row import RoboticsRow
from refiner.utils import check_required_dependencies
from refiner.video import VideoFrameArray, VideoSource
from refiner.worker.context import (
    get_active_stage_index,
    get_active_worker_token,
    get_finalized_workers,
)
from refiner.worker.lifecycle import sort_finalized_workers

_DEFAULT_ARRAY_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_INITIAL_CHUNK_ROWS = 1024
_DONE_MARKER_RELPATH = "_refiner/write_zarr.done"
_MERGE_STARTED_MARKER_RELPATH = "_refiner/write_zarr.started"
_PUBLISH_STARTED_MARKER_RELPATH = "_refiner/write_zarr_publish.started"
_PUBLISH_DONE_MARKER_RELPATH = "_refiner/write_zarr_publish.done"


@dataclass
class _ShardStore:
    root: Any
    arrays: dict[str, Any] = field(default_factory=dict)
    row_end: int = 0


@dataclass(frozen=True)
class _PartStore:
    relpath: str
    paths: set[str]


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
        reduce_to_single_store: bool = True,
        overwrite: bool = True,
    ):
        check_required_dependencies("write_zarr", ["zarr"], dist="zarr")
        if video_frame_batch_size <= 0:
            raise ValueError("video_frame_batch_size must be greater than zero")
        if array_chunk_bytes <= 0:
            raise ValueError("array_chunk_bytes must be greater than zero")
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
        self.reduce_to_single_store = reduce_to_single_store
        self.overwrite = overwrite
        self._stores: dict[str, _ShardStore] = {}
        self._default_arrays: dict[str, str] | None = None
        self._checked_no_overwrite = False
        self._cleared_publish_markers = False
        self._cleared_merge_marker = False

    def write_shard_block(self, shard_id: str, block: Block) -> int:
        count = 0
        pending_store: _ShardStore | None = None
        pending_arrays: dict[str, list[np.ndarray]] = {}
        pending_lengths: list[int] = []
        pending_bytes = 0

        def flush_pending() -> None:
            nonlocal pending_store, pending_arrays, pending_lengths, pending_bytes
            if pending_store is None or not pending_arrays:
                return
            store = pending_store
            rollback_lengths: dict[str, int | None] = {}
            previous_row_end = store.row_end
            for zarr_path in pending_arrays:
                dataset = store.arrays.get(zarr_path)
                rollback_lengths[zarr_path] = (
                    None if dataset is None else int(dataset.shape[0])
                )
            if self.episode_ends_path is not None:
                dataset = store.arrays.get(self.episode_ends_path)
                rollback_lengths[self.episode_ends_path] = (
                    None if dataset is None else int(dataset.shape[0])
                )
            try:
                combined = {
                    zarr_path: (
                        arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
                    )
                    for zarr_path, arrays in pending_arrays.items()
                }
                for zarr_path, array in combined.items():
                    self._validate_array_append(store, zarr_path, array)
                for zarr_path, array in combined.items():
                    self._append_array(store, zarr_path, array)
                if self.episode_ends_path is not None:
                    row_ends = (
                        np.cumsum(np.asarray(pending_lengths, dtype=np.int64))
                        + store.row_end
                    )
                    self._append_array(store, self.episode_ends_path, row_ends)
                    store.row_end = int(row_ends[-1])
            except Exception:
                for zarr_path, length in rollback_lengths.items():
                    if length is None:
                        self._drop_array(store, zarr_path)
                        continue
                    dataset = store.arrays.get(zarr_path)
                    if dataset is not None:
                        dataset.resize((length, *dataset.shape[1:]))
                store.row_end = previous_row_end
                raise
            finally:
                pending_store = None
                pending_arrays = {}
                pending_lengths = []
                pending_bytes = 0

        for row in block:
            try:
                arrays = self._arrays_for_row(row)
                row_arrays, row_videos, lengths = self._row_values(row, arrays)
            except Exception:
                flush_pending()
                raise

            if row_videos:
                flush_pending()
                self._write_row_values(shard_id, row_arrays, row_videos, lengths)
                count += 1
                continue

            if lengths:
                length = lengths[0]
                if any(item != length for item in lengths):
                    flush_pending()
                    raise ValueError(
                        "Zarr arrays for one row must have matching lengths"
                    )
                row_bytes = sum(array.nbytes for array in row_arrays.values())
                if (
                    pending_arrays
                    and pending_bytes + row_bytes > self.array_chunk_bytes
                ):
                    flush_pending()
                if pending_arrays and len(pending_lengths) >= _MAX_INITIAL_CHUNK_ROWS:
                    flush_pending()
                if pending_arrays and set(row_arrays) != set(pending_arrays):
                    flush_pending()
                if pending_arrays and any(
                    pending_arrays[zarr_path][0].shape[1:] != array.shape[1:]
                    or pending_arrays[zarr_path][0].dtype != array.dtype
                    for zarr_path, array in row_arrays.items()
                ):
                    flush_pending()
                store = self._store(shard_id)
                if pending_store is None:
                    pending_store = store
                for zarr_path, array in row_arrays.items():
                    pending_arrays.setdefault(zarr_path, []).append(array)
                pending_lengths.append(length)
                pending_bytes += row_bytes
            count += 1
        flush_pending()
        return count

    def _write_row_values(
        self,
        shard_id: str,
        row_arrays: dict[str, np.ndarray],
        row_videos: list[tuple[str, VideoSource]],
        lengths: list[int],
    ) -> None:
        store: _ShardStore | None = None
        if row_arrays or row_videos:
            store = self._store(shard_id)
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

    def _row_values(
        self,
        row: Row,
        arrays: Mapping[str, str],
    ) -> tuple[dict[str, np.ndarray], list[tuple[str, VideoSource]], list[int]]:
        row_arrays: dict[str, np.ndarray] = {}
        row_videos: list[tuple[str, VideoSource]] = []
        lengths: list[int] = []
        for zarr_path, source_key in arrays.items():
            value = _row_value(row, source_key)
            if value is None:
                raise ValueError(f"Zarr source value is missing: {source_key}")
            if isinstance(value, VideoSource):
                row_videos.append((zarr_path, value))
                continue
            array = _as_array(value)
            if array.ndim == 0:
                array = array.reshape(1)
            lengths.append(int(array.shape[0]))
            row_arrays[zarr_path] = array
        return row_arrays, row_videos, lengths

    async def _append_video(
        self,
        store: _ShardStore,
        path: str,
        video: VideoSource,
        *,
        expected_length: int | None = None,
    ) -> int:
        if isinstance(video, VideoFrameArray):
            if expected_length is not None and video.frame_count != expected_length:
                raise ValueError("Zarr arrays for one row must have matching lengths")
            if video.frame_count == 0:
                empty = np.asarray(video.frames, dtype=np.uint8)
                self._append_array(store, path, empty[:0])
                return 0
            batch: list[np.ndarray] = []
            batch_limit: int | None = None
            for frame in video.iter_frame_arrays():
                batch.append(frame)
                if batch_limit is None:
                    batch_limit = self._video_batch_limit(frame)
                if len(batch) >= batch_limit:
                    self._append_array(
                        store,
                        path,
                        np.stack(batch, axis=0),
                        chunks=(batch_limit, *frame.shape),
                    )
                    batch.clear()
            if batch:
                self._append_array(
                    store,
                    path,
                    np.stack(batch, axis=0),
                    chunks=(batch_limit or len(batch), *batch[0].shape),
                )
            return video.frame_count

        batch: list[np.ndarray] = []
        batch_limit: int | None = None
        count = 0
        async for frame in video.iter_frames():
            batch.append(frame.frame.to_ndarray(format="rgb24"))
            if batch_limit is None:
                batch_limit = self._video_batch_limit(batch[0])
            if len(batch) >= batch_limit:
                if expected_length is not None and count + len(batch) > expected_length:
                    raise ValueError(
                        "Zarr arrays for one row must have matching lengths"
                    )
                self._append_array(
                    store,
                    path,
                    np.stack(batch, axis=0),
                    chunks=(batch_limit, *batch[0].shape),
                )
                count += len(batch)
                batch.clear()
        if batch:
            if expected_length is not None and count + len(batch) > expected_length:
                raise ValueError("Zarr arrays for one row must have matching lengths")
            self._append_array(
                store,
                path,
                np.stack(batch, axis=0),
                chunks=(batch_limit or len(batch), *batch[0].shape),
            )
            count += len(batch)
        if count == 0:
            raise ValueError("Zarr video source produced no frames")
        if expected_length is not None and count != expected_length:
            raise ValueError("Zarr arrays for one row must have matching lengths")
        return count

    def _video_batch_limit(self, frame: np.ndarray) -> int:
        return min(
            self.video_frame_batch_size,
            _batch_length_for_shape(
                (1, *frame.shape), frame.dtype, self.array_chunk_bytes
            ),
        )

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
        if self.reduce_to_single_store and self.overwrite:
            self._clear_merge_marker_once()
        if not self.overwrite and not self.reduce_to_single_store:
            self._clear_publish_markers_once()
        relpath = self._store_relpath(shard_id)
        store = self._stores.get(relpath)
        if store is not None:
            return store
        mode = "w" if self.overwrite or relpath.startswith("_parts/") else "w-"
        import zarr

        store = _ShardStore(
            zarr.open_group(
                store=_zarr_store(self.output, relpath, mode=mode), mode=mode
            )
        )
        try:
            self.output.rm(self._empty_marker_relpath(shard_id))
        except FileNotFoundError:
            pass
        self._stores[relpath] = store
        return store

    def _check_no_overwrite_output(self) -> None:
        if self.overwrite or self._checked_no_overwrite:
            return
        self._checked_no_overwrite = True
        if _output_has_existing_store(self.output):
            raise ValueError("write_zarr output already exists and overwrite=False")

    def _clear_publish_markers_once(self) -> None:
        if self._cleared_publish_markers:
            return
        self._cleared_publish_markers = True
        for marker in (
            _PUBLISH_STARTED_MARKER_RELPATH,
            _PUBLISH_DONE_MARKER_RELPATH,
        ):
            try:
                self.output.rm(marker)
            except FileNotFoundError:
                pass

    def _clear_merge_marker_once(self) -> None:
        if self._cleared_merge_marker:
            return
        self._cleared_merge_marker = True
        try:
            self.output.rm(_DONE_MARKER_RELPATH)
        except FileNotFoundError:
            pass

    def _store_relpath(self, shard_id: str) -> str:
        relpath = _render_store_relpath(
            self.store_template,
            shard_id=shard_id,
            worker_id=get_active_worker_token(),
        )
        if self.reduce_to_single_store or not self.overwrite:
            return _part_store_relpath(relpath)
        return relpath

    def on_shard_complete(self, shard_id: str) -> None:
        self._check_no_overwrite_output()
        if self.reduce_to_single_store and self.overwrite:
            self._clear_merge_marker_once()
        if not self.overwrite and not self.reduce_to_single_store:
            self._clear_publish_markers_once()
        relpath = self._store_relpath(shard_id)
        if relpath not in self._stores:
            try:
                self.output.rm(relpath, recursive=True)
            except FileNotFoundError:
                pass
            with self.output.open(self._empty_marker_relpath(shard_id), mode="wb"):
                pass
        self._stores.pop(relpath, None)

    def _empty_marker_relpath(self, shard_id: str) -> str:
        return self._store_relpath(shard_id) + ".empty"

    def _append_array(
        self,
        store: _ShardStore,
        path: str,
        array: np.ndarray,
        *,
        chunks: tuple[int, ...] | None = None,
    ) -> None:
        _append_zarr_array(
            store.root,
            store.arrays,
            path,
            array,
            chunks=chunks or _chunk_shape(array, self.array_chunk_bytes),
        )

    def _validate_array_append(
        self,
        store: _ShardStore,
        path: str,
        array: np.ndarray,
    ) -> None:
        dataset = store.arrays.get(path)
        if dataset is None:
            return
        _validate_array_schema(path, dataset, array)

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
                array_chunk_bytes=self.array_chunk_bytes,
                overwrite=self.overwrite,
            )
        if not self.overwrite:
            return _ZarrPublishPartsReducerSink(
                output=self.output,
                store_template=self.store_template,
            )
        return _ZarrCleanupReducerSink(
            output=self.output,
            store_template=self.store_template,
        )


class _ZarrCleanupReducerSink(BaseSink):
    def __init__(self, output: DataFolderLike, *, store_template: str) -> None:
        self.output = DataFolder.resolve(output)
        self.store_template = store_template
        self._cleanup = FileCleanupReducerSink(
            output=self.output,
            filename_template=store_template,
            reducer_name="write_zarr_reduce",
            recursive=True,
        )

    @property
    def counts_output_rows(self) -> bool:
        return False

    def write_shard_block(self, shard_id, block) -> None:
        self._cleanup.write_shard_block(shard_id, block)
        stage_index = get_active_stage_index()
        if stage_index is None or stage_index <= 0:
            raise ValueError(
                "write_zarr_reduce requires an active reducer stage with a prior writer stage"
            )
        relpaths = [
            _render_store_relpath(
                self.store_template,
                shard_id=row.shard_id,
                worker_id=row.worker_token,
            )
            for row in sort_finalized_workers(
                get_finalized_workers(stage_index=stage_index - 1)
            )
        ]
        _validate_zarr_stores(self.output, relpaths)
        _remove_parts(self.output)
        self._clear_root_payload_except(relpaths)

    def _clear_root_payload_except(self, relpaths: Iterable[str]) -> None:
        import zarr

        keep_paths = set(relpaths)
        try:
            root = zarr.open_group(store=_zarr_store(self.output, "", mode="r+"))
        except Exception:
            return

        def clear_group(group: Any, prefix: str = "") -> None:
            group_keys = set(group.group_keys())
            for key in sorted({*group.array_keys(), *group_keys}):
                path = f"{prefix}/{key}" if prefix else key
                if path == "_refiner" or path.startswith("_refiner/"):
                    continue
                if path in keep_paths:
                    continue
                if any(keep_path.startswith(f"{path}/") for keep_path in keep_paths):
                    if key in group_keys:
                        clear_group(group[key], path)
                        continue
                del group[key]
            group.attrs.clear()

        clear_group(root)


class _ZarrPublishPartsReducerSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        store_template: str,
    ) -> None:
        self.output = DataFolder.resolve(output)
        self.store_template = store_template
        self._published = False

    @property
    def counts_output_rows(self) -> bool:
        return False

    def write_shard_block(self, shard_id, block) -> None:
        del shard_id, block
        if self._published:
            return
        self._published = True

        stage_index = get_active_stage_index()
        if stage_index is None or stage_index <= 0:
            raise ValueError(
                "write_zarr_publish requires an active reducer stage with a prior writer stage"
            )

        parts = [
            _render_store_relpath(
                self.store_template,
                shard_id=row.shard_id,
                worker_id=row.worker_token,
            )
            for row in sort_finalized_workers(
                get_finalized_workers(stage_index=stage_index - 1)
            )
        ]
        if not parts:
            if self.output.exists(
                _PUBLISH_DONE_MARKER_RELPATH
            ) and not _output_has_payload(self.output):
                _remove_parts(self.output, best_effort=True)
                return
            if _output_has_existing_store(self.output):
                _remove_parts(self.output, best_effort=True)
                raise ValueError("write_zarr output already exists and overwrite=False")
            with self.output.open(_PUBLISH_DONE_MARKER_RELPATH, mode="wb"):
                pass
            _remove_parts(self.output, best_effort=True)
            return

        if self.output.exists(_PUBLISH_DONE_MARKER_RELPATH):
            _remove_parts(self.output, best_effort=True)
            return

        has_existing_output = _output_has_existing_store(self.output)
        parts_validated = False
        if has_existing_output and self.output.exists(_PUBLISH_STARTED_MARKER_RELPATH):
            _validate_zarr_stores(
                self.output, (_part_store_relpath(relpath) for relpath in parts)
            )
            parts_validated = True
        if has_existing_output:
            if not self.output.exists(_PUBLISH_STARTED_MARKER_RELPATH):
                raise ValueError("write_zarr output already exists and overwrite=False")
            self._remove_publish_targets(parts)

        with self.output.open(_PUBLISH_STARTED_MARKER_RELPATH, mode="wb"):
            pass

        try:
            if not parts_validated:
                _validate_zarr_stores(
                    self.output, (_part_store_relpath(relpath) for relpath in parts)
                )
            for final_relpath in parts:
                part_relpath = _part_store_relpath(final_relpath)
                if not self.output.exists(part_relpath):
                    if self.output.exists(part_relpath + ".empty"):
                        continue
                    raise ValueError(f"Zarr part store is missing: {part_relpath}")
                target_parent = final_relpath.rsplit("/", maxsplit=1)[0]
                if target_parent != final_relpath:
                    self.output.makedirs(target_parent, exist_ok=True)
                self.output.copy(
                    part_relpath,
                    final_relpath,
                    recursive=True,
                    on_error="raise",
                )
        except Exception:
            self._remove_publish_targets(parts)
            raise

        with self.output.open(_PUBLISH_DONE_MARKER_RELPATH, mode="wb"):
            pass
        try:
            self.output.rm(_PUBLISH_STARTED_MARKER_RELPATH)
        except (FileNotFoundError, OSError, ValueError):
            pass
        _remove_parts(self.output, best_effort=True)

    def _remove_publish_targets(self, relpaths: Iterable[str]) -> None:
        for relpath in relpaths:
            try:
                self.output.rm(relpath, recursive=True)
            except FileNotFoundError:
                continue


class _ZarrMergeReducerSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        store_template: str,
        episode_ends_path: str | None,
        array_chunk_bytes: int,
        overwrite: bool,
    ) -> None:
        check_required_dependencies("write_zarr", ["zarr"], dist="zarr")
        self.output = DataFolder.resolve(output)
        self.store_template = store_template
        self.episode_ends_path = episode_ends_path
        self.array_chunk_bytes = array_chunk_bytes
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
                "array_chunk_bytes": self.array_chunk_bytes,
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

        expected_parts = self._expected_parts(stage_index)
        if not expected_parts:
            if self.output.exists(_DONE_MARKER_RELPATH) and not _output_has_payload(
                self.output
            ):
                _remove_parts(self.output, best_effort=True)
                return
            if not self.overwrite and _output_has_existing_store(self.output):
                raise ValueError("write_zarr output already exists and overwrite=False")
            import zarr

            final = zarr.open_group(
                store=_zarr_store(self.output, "", mode="a"),
                mode="a",
            )
            if self.overwrite:
                _clear_final_group(final)
            with self.output.open(_DONE_MARKER_RELPATH, mode="wb"):
                pass
            _remove_parts(self.output, best_effort=True)
            return

        if self.output.exists(_DONE_MARKER_RELPATH):
            _remove_parts(self.output, best_effort=True)
            return

        parts = self._collect_parts(expected_parts)
        if not self.overwrite and _output_has_existing_store(self.output):
            if not self.output.exists(_MERGE_STARTED_MARKER_RELPATH):
                raise ValueError("write_zarr output already exists and overwrite=False")

        import zarr

        final = zarr.open_group(
            store=_zarr_store(self.output, "", mode="a"),
            mode="a",
        )
        if self.overwrite:
            _clear_final_group(final)
        elif self.output.exists(_MERGE_STARTED_MARKER_RELPATH):
            _clear_final_group(final)
        else:
            with self.output.open(_MERGE_STARTED_MARKER_RELPATH, mode="wb"):
                pass

        try:
            row_offset = 0
            arrays: dict[str, Any] = {}
            for part in parts:
                source = zarr.open_group(
                    store=_zarr_store(self.output, part.relpath, mode="r"),
                    mode="r",
                )
                for path in sorted(part.paths):
                    source_array = source[path]
                    if path == self.episode_ends_path:
                        if source_array.shape[0] == 0:
                            continue
                        part_last = row_offset
                        batch_size = _batch_length(
                            source_array,
                            self.array_chunk_bytes,
                        )
                        for start in range(0, int(source_array.shape[0]), batch_size):
                            end = min(int(source_array.shape[0]), start + batch_size)
                            values = np.asarray(source_array[start:end], dtype=np.int64)
                            _append_zarr_array(
                                final,
                                arrays,
                                path,
                                values + row_offset,
                                chunks=getattr(source_array, "chunks", None),
                                compressor=getattr(source_array, "compressor", None),
                            )
                            part_last = int(values[-1])
                        row_offset += part_last
                        continue
                    batch_size = _batch_length(source_array, self.array_chunk_bytes)
                    if source_array.shape[0] == 0:
                        _append_zarr_array(
                            final,
                            arrays,
                            path,
                            np.asarray(source_array[:0]),
                            chunks=getattr(source_array, "chunks", None),
                            compressor=getattr(source_array, "compressor", None),
                        )
                        continue
                    for start in range(0, int(source_array.shape[0]), batch_size):
                        end = min(int(source_array.shape[0]), start + batch_size)
                        _append_zarr_array(
                            final,
                            arrays,
                            path,
                            np.asarray(source_array[start:end]),
                            chunks=getattr(source_array, "chunks", None),
                            compressor=getattr(source_array, "compressor", None),
                        )
        except Exception:
            if not self.overwrite:
                _clear_final_group(final)
            raise

        with self.output.open(_DONE_MARKER_RELPATH, mode="wb"):
            pass
        try:
            self.output.rm(_MERGE_STARTED_MARKER_RELPATH)
        except (FileNotFoundError, OSError, ValueError):
            pass
        _remove_parts(self.output, best_effort=True)
        try:
            if not self.output.ls("_parts"):
                self.output.rmdir("_parts")
        except (FileNotFoundError, OSError, ValueError):
            pass

    def _expected_parts(self, stage_index: int) -> list[str]:
        return [
            self._part_relpath(row.shard_id, row.worker_token)
            for row in sort_finalized_workers(
                get_finalized_workers(stage_index=stage_index - 1),
            )
        ]

    def _part_relpath(self, shard_id: str, worker_token: str) -> str:
        relpath = _render_store_relpath(
            self.store_template,
            shard_id=shard_id,
            worker_id=worker_token,
        )
        return _part_store_relpath(relpath)

    def _collect_parts(self, expected_parts: Iterable[str]) -> list[_PartStore]:
        import zarr

        parts: list[_PartStore] = []
        payload_paths: set[str] | None = None
        schemas: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
        for relpath in expected_parts:
            if not self.output.exists(relpath):
                if self.output.exists(f"{relpath}.empty"):
                    continue
                raise ValueError(f"Zarr part store is missing: {relpath}")
            source = zarr.open_group(
                store=_zarr_store(self.output, relpath, mode="r"),
                mode="r",
            )
            source_paths = set(_iter_array_paths(source))
            if not source_paths:
                continue
            source_payload_paths = {
                path for path in source_paths if path != self.episode_ends_path
            }
            if (
                self.episode_ends_path is not None
                and source_payload_paths
                and self.episode_ends_path not in source_paths
            ):
                raise ValueError(
                    f"Zarr part stores must contain {self.episode_ends_path!r}"
                )
            if payload_paths is None:
                payload_paths = source_payload_paths
            elif source_payload_paths != payload_paths:
                raise ValueError(
                    "Zarr part stores must contain the same payload arrays"
                )
            for path in source_paths:
                source_array = source[path]
                schema = (tuple(source_array.shape[1:]), np.dtype(source_array.dtype))
                previous = schemas.setdefault(path, schema)
                if previous != schema:
                    if previous[0] != schema[0]:
                        raise ValueError(
                            f"Zarr arrays for {path!r} must have matching trailing shapes"
                        )
                    raise ValueError(
                        f"Zarr arrays for {path!r} must have matching dtypes"
                    )
            parts.append(_PartStore(relpath=relpath, paths=source_paths))
        return parts


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
    for path in arrays:
        _validate_public_zarr_path(path, "Zarr array path")
    if episode_ends_path is not None:
        _validate_public_zarr_path(episode_ends_path, "episode_ends_path")
    if episode_ends_path is not None and episode_ends_path in arrays:
        raise ValueError(
            f"Zarr array path collides with episode_ends_path: {episode_ends_path}"
        )


def _validate_store_template(store_template: str) -> None:
    _validate_public_zarr_path(store_template, "store_template")
    fields: set[str] = set()
    for _literal_text, field_name, format_spec, conversion in Formatter().parse(
        store_template
    ):
        if field_name is None:
            continue
        if conversion is not None or format_spec:
            raise ValueError(
                "store_template only supports plain {shard_id} and {worker_id} fields"
            )
        if field_name not in {"shard_id", "worker_id"}:
            raise ValueError(
                "store_template only supports plain {shard_id} and {worker_id} fields"
            )
        fields.add(field_name)
    missing_fields = {"shard_id", "worker_id"}.difference(fields)
    if missing_fields:
        raise ValueError(
            "store_template requires fields: "
            + ", ".join(f"{{{field_name}}}" for field_name in sorted(missing_fields))
        )


def _render_store_relpath(
    store_template: str,
    *,
    shard_id: str,
    worker_id: str,
) -> str:
    relpath = store_template.format(shard_id=shard_id, worker_id=worker_id)
    _validate_public_zarr_path(relpath, "rendered store path")
    return relpath


def _validate_public_zarr_path(path: str, label: str) -> None:
    path = str(path)
    if path.startswith("/"):
        raise ValueError(f"{label} must be relative")
    parts = [part for part in path.split("/") if part]
    if any(part in {".", ".."} for part in parts):
        raise ValueError(f"{label} must not contain '.' or '..' segments")
    root = parts[0] if parts else ""
    if root in {"_parts", "_refiner"}:
        raise ValueError(f"{label} must not use reserved root: {root}")


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
    return f"_parts/{relpath}"


def _zarr_store(output: DataFolder, path: str = "", *, mode: str = "r"):
    import zarr

    return zarr.storage.FSStore(
        output._join(path),
        fs=output.fs,
        mode=mode,
        create=mode in {"w", "w-", "a"},
    )


def _iter_array_paths(group: Any, prefix: str = "") -> Iterable[str]:
    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if hasattr(item, "shape"):
            yield path
        else:
            yield from _iter_array_paths(item, path)


def _remove_parts(output: DataFolder, *, best_effort: bool = False) -> None:
    try:
        output.rm("_parts", recursive=True)
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        if not best_effort:
            raise


def _validate_zarr_stores(output: DataFolder, relpaths: Iterable[str]) -> None:
    import zarr

    payload_paths: set[str] | None = None
    schemas: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
    for relpath in relpaths:
        if not output.exists(relpath):
            if output.exists(f"{relpath}.empty"):
                continue
            raise ValueError(f"Zarr store is missing: {relpath}")
        source = zarr.open_group(
            store=_zarr_store(output, relpath, mode="r"),
            mode="r",
        )
        source_paths = set(_iter_array_paths(source))
        if payload_paths is None:
            payload_paths = source_paths
        elif source_paths != payload_paths:
            raise ValueError("Zarr stores must contain the same arrays")
        for path in source_paths:
            source_array = source[path]
            schema = (tuple(source_array.shape[1:]), np.dtype(source_array.dtype))
            previous = schemas.setdefault(path, schema)
            if previous != schema:
                if previous[0] != schema[0]:
                    raise ValueError(
                        f"Zarr arrays for {path!r} must have matching trailing shapes"
                    )
                raise ValueError(f"Zarr arrays for {path!r} must have matching dtypes")


def _clear_final_group(group: Any) -> None:
    for key in sorted({*group.array_keys(), *group.group_keys()}):
        if key != "_parts":
            del group[key]
    group.attrs.clear()


def _group_has_payload(group: Any) -> bool:
    return bool(group.attrs) or any(
        key != "_parts" for key in {*group.array_keys(), *group.group_keys()}
    )


def _output_has_payload(output: DataFolder) -> bool:
    import zarr

    try:
        entries = output.ls("", detail=False)
    except FileNotFoundError:
        return False
    non_part_entries = [
        entry
        for entry in entries
        if str(entry).split("/", maxsplit=1)[0] not in {"_parts", "_refiner"}
    ]
    if not non_part_entries:
        return False
    try:
        group = zarr.open_group(store=_zarr_store(output, "", mode="r"), mode="r")
    except Exception:
        return True
    return _group_has_payload(group)


def _output_has_existing_store(output: DataFolder) -> bool:
    if _output_has_payload(output):
        return True
    if output.exists(_DONE_MARKER_RELPATH) or output.exists(
        _PUBLISH_DONE_MARKER_RELPATH
    ):
        return True
    try:
        entries = output.ls("", detail=False)
    except FileNotFoundError:
        return False
    for entry in entries:
        root = str(entry).split("/", maxsplit=1)[0]
        if root in {"_parts", "_refiner", ".zgroup", ".zattrs", ".zmetadata"}:
            continue
        return True
    return False


def _chunk_shape(array: np.ndarray, target_bytes: int) -> tuple[int, ...]:
    chunk_rows = min(
        _batch_length(array, target_bytes),
        max(int(array.shape[0]), _MAX_INITIAL_CHUNK_ROWS),
    )
    return (chunk_rows, *array.shape[1:])


def _batch_length(array: Any, target_bytes: int) -> int:
    return _batch_length_for_shape(tuple(array.shape), array.dtype, target_bytes)


def _batch_length_for_shape(
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type[Any],
    target_bytes: int,
) -> int:
    dtype = np.dtype(dtype)
    row_values = int(np.prod(shape[1:], dtype=np.int64))
    row_bytes = max(1, dtype.itemsize * max(1, row_values))
    return max(1, target_bytes // row_bytes)


def _validate_array_schema(path: str, dataset: Any, values: np.ndarray) -> None:
    if tuple(dataset.shape[1:]) != tuple(values.shape[1:]):
        raise ValueError(f"Zarr arrays for {path!r} must have matching trailing shapes")
    if dataset.dtype != values.dtype:
        raise ValueError(f"Zarr arrays for {path!r} must have matching dtypes")


def _append_zarr_array(
    root: Any,
    arrays: dict[str, Any],
    path: str,
    values: np.ndarray,
    *,
    chunks: tuple[int, ...] | None = None,
    compressor: Any = None,
) -> None:
    dataset = arrays.get(path)
    if dataset is None:
        dataset = root.create_dataset(
            path,
            shape=(0, *values.shape[1:]),
            chunks=chunks,
            dtype=values.dtype,
            compressor=compressor,
        )
        arrays[path] = dataset
    else:
        _validate_array_schema(path, dataset, values)
    dataset.append(values, axis=0)


__all__ = ["ZarrSink"]
