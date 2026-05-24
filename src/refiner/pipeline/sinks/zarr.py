from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from string import Formatter
from typing import Any

import numpy as np
import pyarrow as pa

from refiner.execution.asyncio.runtime import submit
from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.row import Row
from refiner.pipeline.sinks.base import BaseSink
from refiner.robotics.row import RoboticsRow
from refiner.utils import check_required_dependencies
from refiner.video import VideoFrameArray, VideoSource
from refiner.worker.context import get_active_worker_token

_DEFAULT_ARRAY_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_INITIAL_CHUNK_ROWS = 1024


@dataclass
class _ZarrWriteState:
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
        reduce_to_single_store: bool = True,
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
        self._stores: dict[str, _ZarrWriteState] = {}
        self._default_arrays: dict[str, str] | None = None

    def write_shard_block(self, shard_id: str, block: Block) -> int:
        count = 0
        pending_store: _ZarrWriteState | None = None
        pending_arrays: dict[str, list[np.ndarray]] = {}
        pending_lengths: list[int] = []
        pending_bytes = 0

        def flush_pending() -> None:
            nonlocal pending_store, pending_arrays, pending_lengths, pending_bytes
            if pending_store is None or not pending_arrays:
                return
            store = pending_store
            previous_row_end = store.row_end
            rollback_lengths = self._snapshot_array_lengths(store, pending_arrays)
            if self.episode_ends_path is not None:
                rollback_lengths.update(
                    self._snapshot_array_lengths(store, [self.episode_ends_path])
                )
            try:
                combined = {
                    zarr_path: (
                        arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
                    )
                    for zarr_path, arrays in pending_arrays.items()
                }
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
                self._restore_array_lengths(store, rollback_lengths)
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

            try:
                length = _matching_length(lengths)
            except Exception:
                flush_pending()
                raise
            if length is not None:
                row_bytes = sum(array.nbytes for array in row_arrays.values())
                if pending_arrays and (
                    pending_bytes + row_bytes > self.array_chunk_bytes
                    or len(pending_lengths) >= _MAX_INITIAL_CHUNK_ROWS
                    or set(row_arrays) != set(pending_arrays)
                    or any(
                        pending_arrays[zarr_path][0].shape[1:] != array.shape[1:]
                        or pending_arrays[zarr_path][0].dtype != array.dtype
                        for zarr_path, array in row_arrays.items()
                    )
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
        store = self._store(shard_id)
        expected_length = _matching_length(lengths)

        previous_row_end = store.row_end
        rollback_lengths = self._snapshot_array_lengths(
            store,
            [*row_arrays, *(path for path, _ in row_videos)],
        )
        try:
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
            length = _matching_length(lengths)
            if length is not None and self.episode_ends_path is not None:
                rollback_lengths.update(
                    self._snapshot_array_lengths(store, [self.episode_ends_path])
                )
                row_end = store.row_end + length
                self._append_array(
                    store,
                    self.episode_ends_path,
                    np.asarray([row_end], dtype=np.int64),
                )
                store.row_end = row_end
        except Exception:
            self._restore_array_lengths(store, rollback_lengths)
            store.row_end = previous_row_end
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
        store: _ZarrWriteState,
        path: str,
        video: VideoSource,
        *,
        expected_length: int | None = None,
    ) -> int:
        batch: list[np.ndarray] = []
        batch_limit: int | None = None
        count = 0

        def append_frame(frame: np.ndarray) -> None:
            nonlocal batch_limit
            batch.append(frame)
            if batch_limit is None:
                batch_limit = min(
                    self.video_frame_batch_size,
                    _batch_length_for_shape(
                        (1, *frame.shape), frame.dtype, self.array_chunk_bytes
                    ),
                )

        def flush_batch() -> None:
            nonlocal count
            if not batch:
                return
            if expected_length is not None and count + len(batch) > expected_length:
                raise ValueError("Zarr arrays for one row must have matching lengths")
            self._append_array(
                store,
                path,
                np.stack(batch, axis=0),
                chunks=(batch_limit or len(batch), *batch[0].shape),
            )
            count += len(batch)
            batch.clear()

        if isinstance(video, VideoFrameArray):
            if expected_length is not None and video.frame_count != expected_length:
                raise ValueError("Zarr arrays for one row must have matching lengths")
            if video.frame_count == 0:
                empty = np.asarray(video.frames, dtype=np.uint8)
                self._append_array(store, path, empty[:0])
                return 0
            for frame in video.iter_frame_arrays():
                append_frame(frame)
                if batch_limit is not None and len(batch) >= batch_limit:
                    flush_batch()
            flush_batch()
            return count

        async for frame in video.iter_frames():
            append_frame(frame.frame.to_ndarray(format="rgb24"))
            if batch_limit is not None and len(batch) >= batch_limit:
                flush_batch()
        flush_batch()
        if count == 0:
            raise ValueError("Zarr video source produced no frames")
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

    def _store(self, shard_id: str) -> _ZarrWriteState:
        relpath = self._store_relpath(shard_id)
        store = self._stores.get(relpath)
        if store is not None:
            return store
        import zarr

        store = _ZarrWriteState(
            zarr.open_group(store=_zarr_store(self.output, relpath, mode="w"), mode="w")
        )
        self._stores[relpath] = store
        return store

    def _store_relpath(self, shard_id: str) -> str:
        relpath = _render_store_relpath(
            self.store_template,
            shard_id=shard_id,
            worker_id=get_active_worker_token(),
        )
        return f"_parts/{relpath}" if self.reduce_to_single_store else relpath

    def on_shard_complete(self, shard_id: str) -> None:
        relpath = self._store_relpath(shard_id)
        if relpath not in self._stores:
            import zarr

            zarr.open_group(store=_zarr_store(self.output, relpath, mode="w"), mode="w")
        self._stores.pop(relpath, None)

    def _append_array(
        self,
        store: _ZarrWriteState,
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

    def _snapshot_array_lengths(
        self,
        store: _ZarrWriteState,
        paths: Iterable[str],
    ) -> dict[str, int | None]:
        return {
            path: None
            if (dataset := store.arrays.get(path)) is None
            else int(dataset.shape[0])
            for path in paths
        }

    def _restore_array_lengths(
        self,
        store: _ZarrWriteState,
        lengths: Mapping[str, int | None],
    ) -> None:
        for path, length in lengths.items():
            if length is None:
                store.arrays.pop(path, None)
                try:
                    del store.root[path]
                except (KeyError, FileNotFoundError):
                    pass
                continue
            dataset = store.arrays.get(path)
            if dataset is not None:
                dataset.resize((length, *dataset.shape[1:]))

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
            },
        )

    def build_reducer(self) -> BaseSink | None:
        from refiner.pipeline.sinks.reducer.zarr import ZarrReducerSink

        return ZarrReducerSink(
            output=self.output,
            store_template=self.store_template,
            episode_ends_path=self.episode_ends_path,
            array_chunk_bytes=self.array_chunk_bytes,
            reduce_to_single_store=self.reduce_to_single_store,
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
        return np.asarray(list(value))
    return np.asarray(value)


def _matching_length(lengths: list[int]) -> int | None:
    if not lengths:
        return None
    length = lengths[0]
    if any(item != length for item in lengths):
        raise ValueError("Zarr arrays for one row must have matching lengths")
    return length


def _zarr_store(output: DataFolder, path: str = "", *, mode: str = "r"):
    import zarr

    return zarr.storage.FSStore(
        output._join(path),
        fs=output.fs,
        mode=mode,
        create=mode in {"w", "w-", "a"},
    )


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
