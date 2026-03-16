from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fsspec.spec import AbstractFileSystem

from refiner.execution.asyncio.window import AsyncWindow
from refiner.io import DataFolder
from refiner.io.datafolder import DataFolderLike, DataFolderSpec
from refiner.media import VideoFile
from refiner.pipeline.data.row import ArrowRowView, Row
from refiner.pipeline.data.shard import FilePartsDescriptor, Shard
from refiner.pipeline.sources.base import SourceUnit
from refiner.pipeline.sources.readers.parquet import ParquetReader
from refiner.pipeline.sources.readers.utils import DEFAULT_TARGET_SHARD_BYTES
from refiner.pipeline.utils.cache.file_cache import get_media_cache

_DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_DEFAULT_VIDEO_PATH = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)
_DEFAULT_EPISODES_GLOB_ROOT = "meta/episodes"
_INFO_JSON = "meta/info.json"
_STATS_JSON = "meta/stats.json"
LEROBOT_STATS = "lerobot_stats"
LEROBOT_INFO = "lerobot_info"
LEROBOT_EPISODE_STATS = "lerobot_episode_stats"
_ROW_DROP_PREFIXES = ("stats/", "videos/", "meta/episodes/", "data/")
_ROW_DROP_KEYS = {"dataset_from_index", "dataset_to_index"}
_DATA_FILE_CACHE_NAME = "lerobot:data_files"


@dataclass(frozen=True, slots=True)
class _FrameFileCacheEntry:
    key: tuple[int, Any, Any]
    table: pa.Table


@dataclass(frozen=True, slots=True)
class _DatasetState:
    root: DataFolder
    stats_metadata: Mapping[str, Any]
    fps: int | None
    video_path_template: str
    data_path_template: str
    video_keys: tuple[str, ...]
    robot_type: str | None


class LeRobotEpisodeReader(ParquetReader):
    """LeRobot reader that shards and reads episode parquet files.

    The inherited parquet planner only applies to the episode metadata files
    under `meta/episodes`. Frame parquet files and videos are loaded later when
    each episode row is materialized.
    """

    name = "read_lerobot"

    def __init__(
        self,
        inputs: DataFolderLike | Sequence[DataFolderLike],
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        limit: int | None = None,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
        arrow_batch_size: int = 65536,
        media_max_in_flight: int = 8,
        media_preserve_order: bool = True,
        split_row_groups: bool = True,
    ) -> None:
        """Create a LeRobot episode reader.

        Args:
            inputs: One or more LeRobot dataset roots.
            limit: Optional cap on emitted episode rows.
            target_shard_bytes: Target approximate byte size for episode-file shards.
            num_shards: Optional explicit number of planned episode-file shards.
            media_max_in_flight: Max concurrent frame/video hydration tasks.
            media_preserve_order: Whether hydrated episode rows keep source order.
            split_row_groups: Whether episode parquet byte spans are refined to
                deterministic row ranges inside a file.
        """
        if limit is not None and limit <= 0:
            raise ValueError("limit must be > 0 when provided")
        if media_max_in_flight <= 0:
            raise ValueError("media_max_in_flight must be > 0")

        roots = self._resolve_roots(inputs, fs=fs, storage_options=storage_options)
        self._roots = roots
        self._limit = int(limit) if limit is not None else None
        self._submitted_episodes = 0
        self._media_max_in_flight = int(media_max_in_flight)
        self._media_preserve_order = bool(media_preserve_order)

        super().__init__(
            inputs=tuple(
                (
                    str(root.abs_paths(_DEFAULT_EPISODES_GLOB_ROOT)),
                    root.fs,
                )
                for root in roots
            ),
            fs=fs,
            storage_options=storage_options,
            recursive=True,
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            arrow_batch_size=arrow_batch_size,
            # Episode parquet files are usually small enough that row-range
            # splitting is a reasonable default when callers opt into it.
            split_row_groups=split_row_groups,
        )

        self._datasets: tuple[_DatasetState, ...] | None = None
        self._frames_cache_lock = asyncio.Lock()
        self._frames_cache_entry: _FrameFileCacheEntry | None = None
        self._episode_index = 0

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        """Read one episode-file shard and hydrate episode rows lazily."""
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)
        part = descriptor.parts[0]
        if self._limit is not None and self._submitted_episodes >= self._limit:
            return

        async_window: AsyncWindow[Row] = AsyncWindow(
            max_in_flight=self._media_max_in_flight,
            preserve_order=self._media_preserve_order,
        )
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
                    yield from async_window.flush()
                    return

                row = ArrowRowView(
                    names=names,
                    columns=columns,
                    index_by_name=index_by_name,
                    row_idx=idx,
                )
                episode_index = max(
                    row.get("episode_index", 0),
                    self._episode_index,
                )
                async_window.submit_blocking(
                    self._build_episode_row(row, part.source_index, episode_index)
                )
                yield from async_window.poll()
                self._submitted_episodes += 1

        yield from async_window.flush()

    async def _build_episode_row(
        self, row: Row, source_index: int, episode_index: int
    ) -> Row:
        dataset = self._get_datasets()[source_index]
        episode_stats = self._extract_episode_stats(row)
        metadata = {
            LEROBOT_INFO: {
                "root": dataset.root.abs_paths(""),
                "fps": dataset.fps,
                "robot_type": dataset.robot_type,
            },
            LEROBOT_STATS: dataset.stats_metadata,
        }
        if episode_stats:
            metadata[LEROBOT_EPISODE_STATS] = episode_stats
        patch: dict[str, Any] = {
            "episode_index": episode_index,
            "metadata": metadata,
        }

        frames = await self._load_episode_frames(row, source_index)
        patch["frames"] = frames

        tasks = row.get("tasks")
        if isinstance(tasks, list) and tasks:
            patch["task"] = tasks[0]

        for video_key in dataset.video_keys:
            video = self._build_video(
                episode=row,
                video_key=video_key,
                source_index=source_index,
            )
            if video is not None:
                patch[video_key] = video

        drop_keys = [
            key
            for key in row.keys()
            if key in _ROW_DROP_KEYS
            or any(key.startswith(prefix) for prefix in _ROW_DROP_PREFIXES)
        ]
        base = row.drop(*drop_keys)
        return base.update(patch)

    def _build_video(
        self,
        *,
        episode: Mapping[str, Any],
        video_key: str,
        source_index: int,
    ) -> VideoFile | None:
        dataset = self._get_datasets()[source_index]
        chunk_key = f"videos/{video_key}/chunk_index"
        file_key = f"videos/{video_key}/file_index"
        chunk = episode.get(chunk_key)
        file_idx = episode.get(file_key)
        if chunk is None or file_idx is None:
            return None

        rel = _format_chunked_path(
            dataset.video_path_template,
            video_key=video_key,
            chunk=chunk,
            file_idx=file_idx,
        )
        uri = str(dataset.root.abs_paths(rel))
        from_timestamp = episode.get(f"videos/{video_key}/from_timestamp")
        to_timestamp = episode.get(f"videos/{video_key}/to_timestamp")
        if to_timestamp is None:
            return None

        try:
            from_timestamp_s = (
                float(from_timestamp) if from_timestamp is not None else 0.0
            )
        except (TypeError, ValueError):
            from_timestamp_s = 0.0
        try:
            to_timestamp_s = float(to_timestamp)
        except (TypeError, ValueError):
            return None

        return VideoFile(
            uri=uri,
            from_timestamp_s=from_timestamp_s,
            to_timestamp_s=to_timestamp_s,
        )

    async def _load_episode_frames(
        self,
        episode: Mapping[str, Any],
        source_index: int,
    ) -> list[Row]:
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
                "LeRobot episode row has invalid dataset_from_index/dataset_to_index"
            ) from exc

        if from_idx > to_idx:
            raise ValueError(
                "LeRobot episode row has invalid dataset_from_index/dataset_to_index"
            )

        cache_entry = await self._get_cached_frame_table(
            source_index=source_index,
            chunk=chunk,
            file_idx=file_idx,
        )
        episode_table = self._slice_episode_table(
            cache_entry=cache_entry,
            from_idx=from_idx,
            to_idx=to_idx,
        )
        if episode_table.num_rows <= 0:
            return []

        names = tuple(str(name) for name in episode_table.column_names)
        columns = tuple(
            episode_table.column(i) for i in range(episode_table.num_columns)
        )
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
        source_index: int,
        chunk: Any,
        file_idx: Any,
    ) -> _FrameFileCacheEntry:
        dataset = self._get_datasets()[source_index]
        cache_key = (source_index, chunk, file_idx)
        cached_entry = self._frames_cache_entry
        if cached_entry is not None and cached_entry.key == cache_key:
            return cached_entry

        async with self._frames_cache_lock:
            cached_entry = self._frames_cache_entry
            if cached_entry is not None and cached_entry.key == cache_key:
                return cached_entry

            rel = _format_chunked_path(
                dataset.data_path_template,
                video_key=None,
                chunk=chunk,
                file_idx=file_idx,
            )
            data_file = dataset.root.file(rel)

            async with get_media_cache(_DATA_FILE_CACHE_NAME).cached(
                file=data_file
            ) as local_path:
                with open(local_path, "rb") as handle:
                    table = pq.read_table(handle)

            if "index" not in table.schema.names:
                raise ValueError(
                    "LeRobot frame parquet is missing required 'index' column"
                )
            index_col = table.column("index")
            if index_col.null_count > 0:
                raise ValueError("LeRobot frame parquet contains null index values")

            cache_entry = _FrameFileCacheEntry(key=cache_key, table=table)
            self._frames_cache_entry = cache_entry
            return cache_entry

    @staticmethod
    def _resolve_roots(
        inputs: DataFolderLike | Sequence[DataFolderLike],
        *,
        fs: AbstractFileSystem | None,
        storage_options: Mapping[str, Any] | None,
    ) -> tuple[DataFolder, ...]:
        raw_inputs: list[DataFolderLike] = []
        if isinstance(inputs, (str, PathLike, DataFolder)):
            raw_inputs.append(cast(DataFolderLike, inputs))
        elif (
            isinstance(inputs, tuple)
            and len(inputs) == 2
            and isinstance(inputs[1], AbstractFileSystem)
        ):
            raw_inputs.append(cast(DataFolderSpec, inputs))
        else:
            raw_inputs.extend(cast(Sequence[DataFolderLike], inputs))
        roots = tuple(
            DataFolder.resolve(item, fs=fs, storage_options=storage_options)
            for item in raw_inputs
        )
        if not roots:
            raise ValueError("LeRobot reader requires at least one dataset root")
        return roots

    def _get_datasets(self) -> tuple[_DatasetState, ...]:
        datasets = self._datasets
        if datasets is None:
            datasets = tuple(self._load_dataset_state(root) for root in self._roots)
            self._datasets = datasets
        return datasets

    def _load_dataset_state(self, root: DataFolder) -> _DatasetState:
        if not root.fs.isdir(root.path):
            raise ValueError(
                f"LeRobot reader inputs must be dataset folders, got {root.abs_paths('')!r}"
            )
        info = self._load_info(root)
        stats_metadata = self._load_stats(root)
        features = info.get("features") if isinstance(info, Mapping) else {}
        return _DatasetState(
            root=root,
            stats_metadata=stats_metadata,
            fps=(
                int(info["fps"])
                if isinstance(info, Mapping) and info.get("fps") is not None
                else None
            ),
            video_path_template=(
                str(info.get("video_path"))
                if isinstance(info, Mapping) and isinstance(info.get("video_path"), str)
                else _DEFAULT_VIDEO_PATH
            ),
            data_path_template=(
                str(info.get("data_path"))
                if isinstance(info, Mapping) and isinstance(info.get("data_path"), str)
                else _DEFAULT_DATA_PATH
            ),
            video_keys=self._load_video_keys(
                features if isinstance(features, Mapping) else {}
            ),
            robot_type=(
                str(info.get("robot_type"))
                if isinstance(info, Mapping) and info.get("robot_type") is not None
                else None
            ),
        )

    def _load_info(self, root: DataFolder) -> Mapping[str, Any]:
        info_file = root.file(_INFO_JSON)
        with info_file.open("rb") as handle:
            payload = json.loads(handle.read())
        if not isinstance(payload, Mapping):
            raise ValueError("LeRobot info.json must contain a JSON object")
        return dict(payload)

    def _load_stats(self, root: DataFolder) -> Mapping[str, Any]:
        stats_file = root.file(_STATS_JSON)
        if not stats_file.exists():
            return {}
        with stats_file.open("rb") as handle:
            payload = json.loads(handle.read())
        if not isinstance(payload, Mapping):
            return {}
        return dict(payload)

    def _load_video_keys(self, features: Mapping[str, Any]) -> tuple[str, ...]:
        out: list[str] = []
        for key, feature in features.items():
            if not isinstance(key, str) or not isinstance(feature, Mapping):
                continue
            if feature.get("dtype") == "video":
                out.append(key)
        out.sort()
        return tuple(out)

    def _extract_episode_stats(
        self,
        row: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for key in row.keys():
            if not isinstance(key, str) or not key.startswith("stats/"):
                continue
            _, feature, stat_name = key.split("/", 2)
            out.setdefault(feature, {})[stat_name] = row.get(key)
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
    "LEROBOT_EPISODE_STATS",
    "LEROBOT_STATS",
    "LEROBOT_INFO",
]
