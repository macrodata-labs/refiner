from __future__ import annotations

import json
import math
import numbers
import posixpath
import re
from collections.abc import Iterator, Mapping
from typing import Any

from fsspec import AbstractFileSystem, url_to_fs
import pyarrow.parquet as pq

from refiner.ledger.shard import Shard
from refiner.runtime.types import SourceUnit
from refiner.sources.readers.parquet import ParquetReader
from refiner.sources.row import ArrowRowView, Row
from refiner.video import Video


_DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_DEFAULT_VIDEO_PATH = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)
_DEFAULT_EPISODES_GLOB_ROOT = "meta/episodes"
_INFO_JSON = "meta/info.json"
_STATS_JSON = "meta/stats.json"
LEROBOT_RAW_EPISODE_KEY = "__lerobot_episode"
LEROBOT_CONTEXT_KEY = "__lerobot_context"
_ROW_DROP_PREFIXES = ("stats/", "videos/", "meta/episodes/", "data/")
_ROW_DROP_KEYS = {"dataset_from_index", "dataset_to_index"}
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
        decode: bool = False,
        arrow_batch_size: int = 4096,
    ) -> None:
        self._decode = bool(decode)

        if fs is None:
            self._fs, self._root_path = url_to_fs(root, **dict(storage_options or {}))
        else:
            self._fs = fs
            self._root_path = fs._strip_protocol(root)  # type: ignore[attr-defined]
        self._root_uri = self._fs.unstrip_protocol(self._root_path).removeprefix("file://")

        self._fps = self._load_fps()
        self._stats_metadata = self._load_stats_metadata()
        super().__init__(
            inputs=posixpath.join(self._root_path.rstrip("/"), _DEFAULT_EPISODES_GLOB_ROOT),
            fs=self._fs,
            recursive=True,
            arrow_batch_size=arrow_batch_size,
        )

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        if shard.path not in self.files:
            raise ValueError(f"Unknown LeRobot shard path: {shard.path!r}")
        for batch in super().read_shard(shard):
            names = tuple(str(name) for name in batch.schema.names)
            columns = tuple(batch.column(i) for i in range(batch.num_columns))
            index_by_name = {name: i for i, name in enumerate(names)}
            for idx in range(batch.num_rows):
                episode = ArrowRowView(
                    names=names,
                    columns=columns,
                    index_by_name=index_by_name,
                    row_idx=idx,
                )
                yield self._build_episode_row(episode)

    def describe(self) -> dict[str, Any]:
        return {
            "root": self._root_uri,
            "episode_files": len(self.files),
            "fps": self._fps,
        }

    def _build_episode_row(self, episode: Row) -> Row:
        video_keys = sorted(
            {
                key.removeprefix("videos/").split("/", 1)[0]
                for key in episode
                if key.startswith("videos/") and key.endswith("/chunk_index")
            }
        )

        drop_keys = [
            key
            for key in episode
            if key in _ROW_DROP_KEYS
            or any(key.startswith(prefix) for prefix in _ROW_DROP_PREFIXES)
        ]
        row = episode.drop(*drop_keys) if drop_keys else episode

        row_metadata: dict[str, Any] = (
            dict(self._stats_metadata)
            if isinstance(self._stats_metadata, Mapping)
            else {}
        )
        row_metadata["x"] = {
            LEROBOT_RAW_EPISODE_KEY: episode,
            LEROBOT_CONTEXT_KEY: {
                "root_uri": self._root_uri,
                "video_path_template": _DEFAULT_VIDEO_PATH,
                "video_keys": tuple(video_keys),
                "fps": self._fps,
                "decode": self._decode,
            },
        }
        patch: dict[str, Any] = {"metadata": row_metadata}

        frames = self._load_episode_frames(episode)
        patch["frames"] = frames

        tasks = row.get("tasks")
        if isinstance(tasks, list) and tasks:
            patch["task"] = tasks[0]

        for video_key in video_keys:
            video = self._build_video(
                episode=episode, frames=frames, video_key=video_key
            )
            if video is not None:
                patch[video_key] = video

        return row.update(patch)

    def _build_video(
        self,
        *,
        episode: Mapping[str, Any],
        frames: list[dict[str, Any]],
        video_key: str,
    ) -> Video | None:
        chunk_raw = episode.get(f"videos/{video_key}/chunk_index")
        file_idx_raw = episode.get(f"videos/{video_key}/file_index")
        if chunk_raw is None or file_idx_raw is None:
            return None
        chunk = _as_int(chunk_raw)
        file_idx = _as_int(file_idx_raw)
        if chunk is None or file_idx is None:
            return None

        rel = str(
            _DEFAULT_VIDEO_PATH.format(
                video_key=video_key, chunk_index=chunk, file_index=file_idx
            )
        )
        uri = posixpath.join(self._root_uri.rstrip("/"), rel)
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

        return Video(
            uri=uri,
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
            bytes=None,
            decode=self._decode,
        )

    def _load_episode_frames(self, episode: Mapping[str, Any]) -> list[dict[str, Any]]:
        chunk_raw = episode.get("data/chunk_index")
        file_idx_raw = episode.get("data/file_index")
        if chunk_raw is None or file_idx_raw is None:
            return []
        chunk = _as_int(chunk_raw)
        file_idx = _as_int(file_idx_raw)
        if chunk is None or file_idx is None:
            return []

        rel = str(_DEFAULT_DATA_PATH.format(chunk_index=chunk, file_index=file_idx))
        data_path = posixpath.join(self._root_path.rstrip("/"), rel)
        with self._fs.open(data_path, mode="rb") as f:
            rows = pq.read_table(f).to_pylist()
        if not rows:
            return []

        from_idx_raw = episode.get("dataset_from_index")
        to_idx_raw = episode.get("dataset_to_index")
        from_idx = _as_int(from_idx_raw)
        to_idx = _as_int(to_idx_raw)
        if from_idx is not None and to_idx is not None:
            out_rows: list[dict[str, Any]] = []
            for row in rows:
                idx_raw = row.get("index")
                idx = _as_int(idx_raw)
                if idx is None:
                    continue
                if from_idx <= idx < to_idx:
                    out_rows.append(dict(row))
            return out_rows

        episode_index_raw = episode.get("episode_index")
        episode_index = _as_int(episode_index_raw)
        if episode_index is None:
            return []

        out = []
        for row in rows:
            value = row.get("episode_index")
            row_episode_index = _as_int(value)
            if row_episode_index is None:
                continue
            if row_episode_index == episode_index:
                out.append(dict(row))

        out.sort(key=lambda row: _as_int(row.get("index")) or 0)
        return out

    def _load_fps(self) -> int | None:
        info_path = posixpath.join(self._root_path.rstrip("/"), _INFO_JSON)
        if not self._fs.exists(info_path):
            return None
        with self._fs.open(info_path, mode="rb") as f:
            payload = json.loads(f.read())
        if not isinstance(payload, Mapping):
            return None

        return _as_int(payload.get("fps"))

    def _load_stats_metadata(self) -> Mapping[str, Any] | None:
        stats_path = posixpath.join(self._root_path.rstrip("/"), _STATS_JSON)
        if self._fs.exists(stats_path):
            with self._fs.open(stats_path, mode="rb") as f:
                payload = json.loads(f.read())
            if isinstance(payload, Mapping):
                return dict(payload)
        return None


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


__all__ = [
    "LeRobotEpisodeReader",
    "LEROBOT_RAW_EPISODE_KEY",
    "LEROBOT_CONTEXT_KEY",
]
