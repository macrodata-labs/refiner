from __future__ import annotations

import asyncio
import json
import posixpath
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from fsspec.spec import AbstractFileSystem
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from refiner.execution.asyncio.window import AsyncWindow
from refiner.io import DataFile
from refiner.media import MediaFile, Video, get_media_cache
from refiner.pipeline.data.row import ArrowRowView, DictRow, Row
from refiner.pipeline.data.shard import Shard
from refiner.pipeline.sources.base import SourceUnit
from refiner.pipeline.sources.readers.parquet import ParquetReader


_DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_DEFAULT_VIDEO_PATH = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)
_DEFAULT_EPISODES_GLOB_ROOT = "meta/episodes"
_INFO_JSON = "meta/info.json"
_STATS_JSON = "meta/stats.json"
LEROBOT_STATS = "lerobot_stats"
LEROBOT_INFO = "lerobot_info"
_ROW_DROP_PREFIXES = ("stats/", "videos/", "meta/episodes/", "data/")
_ROW_DROP_KEYS = {"dataset_from_index", "dataset_to_index"}
_DATA_FILE_CACHE_NAME = "lerobot:data_files"


@dataclass(frozen=True, slots=True)
class _FrameFileCacheEntry:
    key: tuple[Any, Any]
    table: pa.Table


class LeRobotEpisodeReader(ParquetReader):
    """Episode-granular LeRobot source."""

    name = "read_lerobot"

    def __init__(
        self,
        root: str,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        decode: Literal[True, False, None] = None,
        limit: int | None = None,
        arrow_batch_size: int = 65536,
    ) -> None:
        if decode is not None and decode not in (True, False):
            raise ValueError("decode must be True, False, or None")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be > 0 when provided")

        # Keep both public `root` and private `_root` fields so cloud-submitted
        # pickles remain readable by older worker images that still reference `_root`.
        self.root = root.rstrip("/")
        self._root = self.root
        self._limit = int(limit) if limit is not None else None
        self._submitted_episodes = 0
        self._decode = decode

        super().__init__(
            inputs=posixpath.join(self.root, _DEFAULT_EPISODES_GLOB_ROOT),
            fs=fs,
            storage_options=storage_options,
            recursive=True,
            arrow_batch_size=arrow_batch_size,
        )

        info = self._load_info(fs=fs, storage_options=storage_options)
        self._stats_metadata = self._load_stats(fs=fs, storage_options=storage_options)
        self._fps = (
            int(info["fps"])
            if isinstance(info, Mapping) and info.get("fps") is not None
            else None
        )
        self._video_path_template = (
            str(info.get("video_path"))
            if isinstance(info, Mapping) and isinstance(info.get("video_path"), str)
            else _DEFAULT_VIDEO_PATH
        )
        self._data_path_template = (
            str(info.get("data_path"))
            if isinstance(info, Mapping) and isinstance(info.get("data_path"), str)
            else _DEFAULT_DATA_PATH
        )
        features = info.get("features") if isinstance(info, Mapping) else {}
        self._features = features if isinstance(features, Mapping) else {}
        self._video_keys = self._load_video_keys()
        self._robot_type = (
            str(info.get("robot_type"))
            if isinstance(info, Mapping) and info.get("robot_type") is not None
            else None
        )
        self._frames_cache_lock = asyncio.Lock()
        self._frames_cache_entry: _FrameFileCacheEntry | None = None

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        if shard.path not in self.files:
            raise ValueError(f"Unknown LeRobot shard path: {shard.path!r}")
        if self._limit is not None and self._submitted_episodes >= self._limit:
            return

        async_window: AsyncWindow[Row] = AsyncWindow(max_in_flight=8, preserve_order=True)
        for batch in super().read_shard(shard):
            if not isinstance(batch, (pa.RecordBatch, pa.Table)):
                raise TypeError(
                    "LeRobotEpisodeReader requires Arrow batches from ParquetReader"
                )

            names = tuple(str(name) for name in batch.schema.names)
            columns = tuple(batch.column(i) for i in range(batch.num_columns))
            index_by_name = {name: i for i, name in enumerate(names)}
            for idx in range(batch.num_rows):
                if self._limit is not None and self._submitted_episodes >= self._limit:
                    yield from async_window.drain(flush=True)
                    return

                row = ArrowRowView(
                    names=names,
                    columns=columns,
                    index_by_name=index_by_name,
                    row_idx=idx,
                )
                async_window.submit(self._build_episode_row(row))
                self._submitted_episodes += 1
                yield from async_window.drain(flush=False)

        yield from async_window.drain(flush=True)

    def describe(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "episode_files": len(self.files),
            "fps": self._fps,
            "limit": self._limit,
        }

    async def _build_episode_row(self, row: Row) -> Row:
        metadata = {
            LEROBOT_INFO: {
                "root": self.root,
                "fps": self._fps,
                "robot_type": self._robot_type,
            },
            LEROBOT_STATS: self._stats_metadata,
        }
        patch: dict[str, Any] = {"metadata": metadata}

        frames = await self._load_episode_frames(row)
        patch["frames"] = frames

        tasks = row.get("tasks")
        if isinstance(tasks, list) and tasks:
            patch["task"] = tasks[0]

        for video_key in self._video_keys:
            video = self._build_video(episode=row, frames=frames, video_key=video_key)
            if video is not None:
                patch[video_key] = video

        drop_keys = [
            key
            for key in row.keys()
            if key in _ROW_DROP_KEYS
            or any(key.startswith(prefix) for prefix in _ROW_DROP_PREFIXES)
        ]
        base = row.drop(*drop_keys)
        data = base.to_dict()
        data.update(patch)
        return DictRow(data=data, metadata=metadata)

    def _build_video(
        self,
        *,
        episode: Mapping[str, Any],
        frames: list[Row],
        video_key: str,
    ) -> Video | None:
        chunk_key = f"videos/{video_key}/chunk_index"
        file_key = f"videos/{video_key}/file_index"
        chunk = episode.get(chunk_key)
        file_idx = episode.get(file_key)
        if chunk is None or file_idx is None:
            return None

        rel = _format_chunked_path(
            self._video_path_template,
            video_key=video_key,
            chunk=chunk,
            file_idx=file_idx,
        )
        uri = posixpath.join(self.root, rel)
        first = frames[0] if frames else {}
        from_timestamp = episode.get(f"videos/{video_key}/from_timestamp")
        to_timestamp = episode.get(f"videos/{video_key}/to_timestamp")
        if to_timestamp is None:
            return None

        episode_index = episode.get("episode_index")
        frame_index = first.get("frame_index") if isinstance(first, Mapping) else None
        timestamp = first.get("timestamp") if isinstance(first, Mapping) else None
        try:
            episode_index = int(episode_index) if episode_index is not None else None
        except (TypeError, ValueError):
            episode_index = None
        try:
            frame_index = int(frame_index) if frame_index is not None else None
        except (TypeError, ValueError):
            frame_index = None
        try:
            timestamp = float(timestamp) if timestamp is not None else None
        except (TypeError, ValueError):
            timestamp = None

        try:
            from_timestamp = (
                float(from_timestamp) if from_timestamp is not None else None
            )
        except (TypeError, ValueError):
            from_timestamp = None
        try:
            to_timestamp = float(to_timestamp) if to_timestamp is not None else None
        except (TypeError, ValueError):
            to_timestamp = None

        if self._decode is None and (
            from_timestamp is not None or to_timestamp is not None
        ):
            raise ValueError(
                "decode is None cannot read timestamped videos; "
                "pass decode=True to materialize clip-aligned bytes."
            )

        return Video(
            media=MediaFile(uri),
            video_key=video_key,
            relative_path=rel,
            episode_index=episode_index,
            frame_index=frame_index,
            timestamp_s=timestamp,
            from_timestamp_s=from_timestamp,
            to_timestamp_s=to_timestamp,
            chunk_index=chunk,
            file_index=file_idx,
            fps=self._fps,
        )

    async def _load_episode_frames(self, episode: Mapping[str, Any]) -> list[Row]:
        chunk = episode["data/chunk_index"]
        file_idx = episode["data/file_index"]
        if chunk is None or file_idx is None:
            return []

        if "dataset_from_index" not in episode or "dataset_to_index" not in episode:
            raise ValueError(
                "LeRobot episode row is missing required "
                "dataset_from_index/dataset_to_index"
            )
        from_idx = episode["dataset_from_index"]
        to_idx = episode["dataset_to_index"]

        try:
            from_idx = int(from_idx)
            to_idx = int(to_idx)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "LeRobot episode row has invalid "
                "dataset_from_index/dataset_to_index"
            ) from exc

        if from_idx > to_idx:
            raise ValueError(
                "LeRobot episode row has invalid "
                "dataset_from_index/dataset_to_index"
            )

        cache_entry = await self._get_cached_frame_table(chunk=chunk, file_idx=file_idx)
        episode_table = self._slice_episode_table(
            cache_entry=cache_entry,
            from_idx=from_idx,
            to_idx=to_idx,
        )
        if episode_table.num_rows <= 0:
            return []

        names = tuple(str(name) for name in episode_table.column_names)
        columns = tuple(episode_table.column(i) for i in range(episode_table.num_columns))
        index_by_name = {name: i for i, name in enumerate(names)}

        return [
            ArrowRowView(
                names=names,
                columns=columns,
                index_by_name=index_by_name,
                row_idx=row_idx,
            )
            for row_idx in range(episode_table.num_rows)
        ]

    def _slice_episode_table(
        self,
        *,
        cache_entry: _FrameFileCacheEntry,
        from_idx: int,
        to_idx: int,
    ) -> pa.Table:
        table = cache_entry.table
        if table.num_rows <= 0:
            return table.slice(0, 0)

        index_expr = pc.field("index")
        mask = (index_expr >= pa.scalar(from_idx, type=pa.int64())) & (
            index_expr < pa.scalar(to_idx, type=pa.int64())
        )
        return table.filter(mask)

    async def _get_cached_frame_table(
        self,
        *,
        chunk: Any,
        file_idx: Any,
    ) -> _FrameFileCacheEntry:
        cache_key = (chunk, file_idx)
        cached_entry = self._frames_cache_entry
        if cached_entry is not None and cached_entry.key == cache_key:
            return cached_entry

        async with self._frames_cache_lock:
            cached_entry = self._frames_cache_entry
            if cached_entry is not None and cached_entry.key == cache_key:
                return cached_entry

            rel = _format_chunked_path(
                self._data_path_template,
                video_key=None,
                chunk=chunk,
                file_idx=file_idx,
            )
            data_file = DataFile.resolve(
                posixpath.join(self.root, rel),
                fs=self._fs,
                storage_options=self._storage_options,
            )

            async with get_media_cache(_DATA_FILE_CACHE_NAME).cached(file=data_file) as local_path:
                with open(local_path, "rb") as handle:
                    table = pq.read_table(handle)

            if "index" not in table.schema.names:
                raise ValueError("LeRobot frame parquet is missing required 'index' column")
            index_col = table.column("index")
            if index_col.null_count > 0:
                raise ValueError("LeRobot frame parquet contains null index values")

            cache_entry = _FrameFileCacheEntry(key=cache_key, table=table)
            self._frames_cache_entry = cache_entry
            return cache_entry

    def _load_info(
        self,
        *,
        fs: AbstractFileSystem | None,
        storage_options: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        info_file = DataFile.resolve(
            posixpath.join(self.root, _INFO_JSON),
            fs=fs,
            storage_options=storage_options,
        )
        with info_file.open("rb") as handle:
            payload = json.loads(handle.read())
        if not isinstance(payload, Mapping):
            raise ValueError("LeRobot info.json must contain a JSON object")
        return dict(payload)

    def _load_stats(
        self,
        *,
        fs: AbstractFileSystem | None,
        storage_options: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        stats_file = DataFile.resolve(
            posixpath.join(self.root, _STATS_JSON),
            fs=fs,
            storage_options=storage_options,
        )
        if not stats_file.exists():
            return {}
        with stats_file.open("rb") as handle:
            payload = json.loads(handle.read())
        if not isinstance(payload, Mapping):
            return {}
        return dict(payload)

    def _load_video_keys(self) -> list[str]:
        out: list[str] = []
        for key, feature in self._features.items():
            if not isinstance(key, str) or not isinstance(feature, Mapping):
                continue
            if feature.get("dtype") == "video":
                out.append(key)
        out.sort()
        return out


def _format_chunked_path(
    template: str,
    *,
    video_key: str | None,
    chunk: str | int,
    file_idx: int,
) -> str:
    return str(
        template.format(
            video_key=video_key if video_key is not None else "",
            chunk=chunk,
            chunk_key=chunk,
            chunk_index=chunk,
            file=file_idx,
            file_index=file_idx,
        )
    )


__all__ = [
    "LeRobotEpisodeReader",
    "LEROBOT_STATS",
    "LEROBOT_INFO",
]
