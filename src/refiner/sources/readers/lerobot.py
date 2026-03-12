from __future__ import annotations

import asyncio
import json
import math
import numbers
import posixpath
import re
from collections.abc import Iterator, Mapping
from typing import Any, Literal
import pyarrow.compute as pc

from fsspec.spec import AbstractFileSystem
import pyarrow.parquet as pq

from refiner.io import DataFile
from refiner.ledger.shard import Shard
from refiner.media import MediaFile, Video, get_media_cache
from refiner.runtime.types import SourceUnit
from refiner.sources.readers.parquet import ParquetReader
from refiner.sources.row import ArrowRowView, DictRow, Row
from refiner.runtime.execution.async_window import AsyncWindow


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
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


class LeRobotEpisodeReader(ParquetReader):
    """Episode-granular LeRobot source.

    Emits one row per episode, with frame payloads embedded under `frames` and
    per-camera `Video` values in flattened columns.
    """

    name = "read_lerobot"

    def __init__(
        self,
        root: str,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        decode: Literal[True, False, None] = None,
        arrow_batch_size: int = 65536,
    ) -> None:
        if decode is not None and decode not in (True, False):
            raise ValueError("decode must be True, False, or None")
        self.root = root.rstrip("/")
        super().__init__(
            inputs=posixpath.join(self.root, _DEFAULT_EPISODES_GLOB_ROOT),
            fs=fs,
            storage_options=storage_options,
            recursive=True,
            arrow_batch_size=arrow_batch_size,
        )
        self._decode = decode
        info = self._load_info(fs=fs, storage_options=storage_options)
        self._stats_metadata = self._load_stats(fs=fs, storage_options=storage_options)
        self._fps = _as_int(info.get("fps")) if isinstance(info, Mapping) else None
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

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        if shard.path not in self.files:
            raise ValueError(f"Unknown LeRobot shard path: {shard.path!r}")
        async_window = AsyncWindow(max_in_flight=2, preserve_order=True)
        for batch in super().read_shard(shard):
            names = tuple(str(name) for name in batch.schema.names)
            columns = tuple(batch.column(i) for i in range(batch.num_columns))
            index_by_name = {name: i for i, name in enumerate(names)}
            for idx in range(batch.num_rows):
                async_window.submit(self._build_episode_row(
                    ArrowRowView(
                        names=names,
                        columns=columns,
                        index_by_name=index_by_name,
                        row_idx=idx,
                    )
                ))
                yield from async_window.drain(flush=False)

        yield from async_window.drain(flush=True)

    def describe(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "episode_files": len(self.files),
            "fps": self._fps,
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

        frames = await asyncio.get_running_loop().run_in_executor(None, self._load_episode_frames, row)
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
        chunk_raw = episode.get(f"videos/{video_key}/chunk_index")
        file_idx_raw = episode.get(f"videos/{video_key}/file_index")
        if chunk_raw is None or file_idx_raw is None:
            return None
        chunk = _as_chunk_id(chunk_raw)
        file_idx = _as_int(file_idx_raw)
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
        episode_index_raw = episode.get("episode_index")
        frame_index_raw = first.get("frame_index")
        timestamp_raw = first.get("timestamp")
        from_ts_raw = episode.get(f"videos/{video_key}/from_timestamp")
        to_ts_raw = episode.get(f"videos/{video_key}/to_timestamp")

        episode_index = _as_int(episode_index_raw)
        frame_index = _as_int(frame_index_raw)
        timestamp = _as_float(timestamp_raw)
        from_timestamp = _as_float(from_ts_raw)
        to_timestamp = _as_float(to_ts_raw)
        if self._decode is None and (
            from_timestamp is not None or to_timestamp is not None
        ):
            raise ValueError(
                "decode is None cannot read timestamped videos; pass decode=True"
                " to materialize clip-aligned bytes."
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

    def _load_episode_frames(self, episode: Mapping[str, Any]) -> list[Row]:
        chunk_raw = episode.get("data/chunk_index")
        file_idx_raw = episode.get("data/file_index")
        if chunk_raw is None or file_idx_raw is None:
            return []
        chunk = _as_chunk_id(chunk_raw)
        file_idx = _as_int(file_idx_raw)
        if chunk is None or file_idx is None:
            return []

        from_idx_raw = episode.get("dataset_from_index")
        to_idx_raw = episode.get("dataset_to_index")
        from_idx = _as_int(from_idx_raw)
        to_idx = _as_int(to_idx_raw)
        if from_idx is None or to_idx is None:
            raise ValueError(
                "LeRobot episode row is missing required dataset_from_index/dataset_to_index"
            )
        if to_idx < from_idx:
            raise ValueError(
                "LeRobot episode row has invalid bounds: "
                f"dataset_from_index={from_idx}, dataset_to_index={to_idx}"
            )

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

        with get_media_cache(_DATA_FILE_CACHE_NAME).cached(
            file=data_file,
        ) as local_path, open(local_path, "rb") as f:
            file_num_rows = int(pq.ParquetFile(local_path).metadata.num_rows)
            if from_idx < 0 or to_idx > file_num_rows:
                raise ValueError(
                    "LeRobot dataset bounds are out of range for data parquet file: "
                    f"dataset_from_index={from_idx}, "
                    f"dataset_to_index={to_idx}, file_num_rows={file_num_rows}"
                )
            table = pq.read_table(f, filters=(
                (pc.greater_equal(pc.field("index"), from_idx)) & (pc.less(pc.field("index"), to_idx))
            ))

        names = tuple(str(name) for name in table.column_names)
        columns = tuple(table.column(i) for i in range(table.num_columns))
        index_by_name = {name: i for i, name in enumerate(names)}

        return [
            ArrowRowView(
                names=names,
                columns=columns,
                index_by_name=index_by_name,
                row_idx=row_idx,
            )
            for row_idx in range(table.num_rows)
        ]

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
        with info_file.open("rb") as f:
            payload = json.loads(f.read())
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
        with stats_file.open("rb") as f:
            payload = json.loads(f.read())
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


def _filter_index_range(
    table: pq.Table,
    *,
    from_idx: int,
    to_idx: int,
) -> pq.Table:
    if table.num_rows == 0:
        return table
    return table.filter(
        (pc.greater_equal(pc.field("index"), from_idx)) & (pc.less(pc.field("index"), to_idx))
    )


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        float_value = float(value)
        if math.isfinite(float_value) and float_value.is_integer():
            return int(float_value)
    if isinstance(value, str):
        stripped = value.strip()
        return int(stripped) if _INT_RE.match(stripped) else None
    return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        float_value = float(value)
        return float_value if math.isfinite(float_value) else None
    if isinstance(value, str):
        stripped = value.strip()
        if not _FLOAT_RE.match(stripped):
            return None
        float_value = float(stripped)
        return float_value if math.isfinite(float_value) else None
    return None


def _as_chunk_id(value: Any) -> str | int | None:
    if value is None or isinstance(value, bool):
        return None
    numeric = _as_int(value)
    if numeric is not None:
        return numeric
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    return str(value)


def _format_chunked_path(
    template: str,
    *,
    video_key: str | None,
    chunk: str | int,
    file_idx: int,
) -> str:
    kwargs = {
        "video_key": video_key if video_key is not None else "",
        "chunk": chunk,
        "chunk_key": chunk,
        "chunk_index": chunk,
        "file": file_idx,
        "file_index": file_idx,
    }
    try:
        return str(template.format(**kwargs))
    except Exception as e:
        raise ValueError(f"Invalid LeRobot path template {template!r}: {e}") from e


__all__ = [
    "LeRobotEpisodeReader",
    "LEROBOT_STATS",
    "LEROBOT_INFO",
]
