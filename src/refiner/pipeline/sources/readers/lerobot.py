from __future__ import annotations

import json
import posixpath
from collections.abc import Iterator, Mapping
from functools import cached_property
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from refiner.media import MediaFile, Video
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.shard import Shard
from refiner.pipeline.sources.base import SourceUnit
from refiner.pipeline.sources.readers.parquet import ParquetReader


_DEFAULT_DATA_PATH = "data/chunk-{chunk_key}/file-{file_index:03d}.parquet"
_DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_key}/file-{file_index:03d}.mp4"
_DEFAULT_EPISODES_GLOB_ROOT = "meta/episodes"
_INFO_JSON = "meta/info.json"
_STATS_JSON = "meta/stats.json"
LEROBOT_INFO = "lerobot_info"
LEROBOT_STATS = "lerobot_stats"
LEROBOT_RAW_EPISODE_KEY = "__lerobot_episode"
LEROBOT_CONTEXT_KEY = "__lerobot_context"
_ROW_DROP_PREFIXES = ("stats/", "videos/", "meta/episodes/", "data/")
_ROW_DROP_KEYS = {"dataset_from_index", "dataset_to_index"}


class LeRobotEpisodeReader(ParquetReader):
    """Episode-granular LeRobot source."""

    name = "read_lerobot"

    def __init__(
        self,
        root: str,
        *,
        fs=None,
        storage_options: Mapping[str, Any] | None = None,
        decode: bool = False,
        arrow_batch_size: int = 4096,
    ) -> None:
        self._root = root
        self._decode = bool(decode)
        super().__init__(
            inputs=posixpath.join(root.rstrip("/"), _DEFAULT_EPISODES_GLOB_ROOT),
            fs=fs,
            storage_options=storage_options,
            recursive=True,
            arrow_batch_size=arrow_batch_size,
        )

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        if shard.path not in self.files:
            raise ValueError(f"Unknown LeRobot shard path: {shard.path!r}")
        for unit in super().read_shard(shard):
            if not isinstance(unit, (pa.RecordBatch, pa.Table)):
                raise TypeError(
                    "LeRobotEpisodeReader requires Arrow batches from ParquetReader"
                )
            yield from (
                self._build_episode_row(dict(episode)) for episode in unit.to_pylist()
            )

    def describe(self) -> dict[str, Any]:
        return {
            "root": self.root_uri,
            "episode_files": len(self.files),
            "fps": self.fps,
        }

    @cached_property
    def root_path(self) -> str:
        return self.fs._strip_protocol(self._root)  # type: ignore[attr-defined]

    @cached_property
    def root_uri(self) -> str:
        return self.fs.unstrip_protocol(self.root_path).removeprefix("file://")

    @cached_property
    def fps(self) -> int | None:
        payload = self._load_json(_INFO_JSON)
        if not isinstance(payload, Mapping):
            return None
        value = payload.get("fps")
        return None if value is None else int(value)

    @cached_property
    def stats_metadata(self) -> dict[str, Any]:
        payload = self._load_json(_STATS_JSON)
        return dict(payload) if isinstance(payload, Mapping) else {}

    def _build_episode_row(self, episode: dict[str, Any]) -> DictRow:
        video_keys = sorted(
            {
                key.removeprefix("videos/").split("/", 1)[0]
                for key in episode
                if key.startswith("videos/") and key.endswith("/chunk_index")
            }
        )
        row = {
            key: value
            for key, value in episode.items()
            if key not in _ROW_DROP_KEYS
            and not any(key.startswith(prefix) for prefix in _ROW_DROP_PREFIXES)
        }

        row_metadata: dict[str, Any] = {
            **self.stats_metadata,
            LEROBOT_INFO: {
                "root": self.root_uri,
                "fps": self.fps,
                "robot_type": None,
            },
            LEROBOT_STATS: self.stats_metadata,
            "x": {
                LEROBOT_RAW_EPISODE_KEY: dict(episode),
                LEROBOT_CONTEXT_KEY: {
                    "root_uri": self.root_uri,
                    "video_path_template": _DEFAULT_VIDEO_PATH,
                    "video_keys": tuple(video_keys),
                    "fps": self.fps,
                    "decode": self._decode,
                },
            },
        }
        row["metadata"] = row_metadata

        frames = self._load_episode_frames(episode)
        row["frames"] = frames

        tasks = row.get("tasks")
        if isinstance(tasks, list) and tasks:
            row["task"] = tasks[0]

        for video_key in video_keys:
            video = self._build_video(
                episode=episode, frames=frames, video_key=video_key
            )
            if video is not None:
                row[video_key] = video

        return DictRow(row, metadata={})

    def _build_video(
        self,
        *,
        episode: Mapping[str, Any],
        frames: list[dict[str, Any]],
        video_key: str,
    ) -> Video | None:
        chunk_value = episode.get(f"videos/{video_key}/chunk_index")
        file_index_value = episode.get(f"videos/{video_key}/file_index")
        if chunk_value is None or file_index_value is None:
            return None

        chunk_key = str(chunk_value)
        file_index = int(file_index_value)
        relative_path = _DEFAULT_VIDEO_PATH.format(
            video_key=video_key,
            chunk_key=_format_chunk_key(chunk_key),
            file_index=file_index,
        )
        first_frame = frames[0] if frames else {}

        return Video(
            media=MediaFile(posixpath.join(self.root_uri.rstrip("/"), relative_path)),
            video_key=video_key,
            relative_path=relative_path,
            episode_index=(
                None
                if episode.get("episode_index") is None
                else int(episode["episode_index"])
            ),
            frame_index=(
                None
                if first_frame.get("frame_index") is None
                else int(first_frame["frame_index"])
            ),
            timestamp_s=(
                None
                if first_frame.get("timestamp") is None
                else float(first_frame["timestamp"])
            ),
            from_timestamp_s=(
                None
                if episode.get(f"videos/{video_key}/from_timestamp") is None
                else float(episode[f"videos/{video_key}/from_timestamp"])
            ),
            to_timestamp_s=(
                None
                if episode.get(f"videos/{video_key}/to_timestamp") is None
                else float(episode[f"videos/{video_key}/to_timestamp"])
            ),
            chunk_index=chunk_key,
            file_index=file_index,
            fps=self.fps,
            decode=self._decode,
        )

    def _load_episode_frames(self, episode: Mapping[str, Any]) -> list[dict[str, Any]]:
        chunk_value = episode.get("data/chunk_index")
        file_index_value = episode.get("data/file_index")
        if chunk_value is None or file_index_value is None:
            return []

        relative_path = _DEFAULT_DATA_PATH.format(
            chunk_key=_format_chunk_key(str(chunk_value)),
            file_index=int(file_index_value),
        )
        data_path = posixpath.join(self.root_path.rstrip("/"), relative_path)
        with self.fs.open(data_path, mode="rb") as handle:
            rows = pq.read_table(handle).to_pylist()
        if not rows:
            return []

        dataset_from = (
            None
            if episode.get("dataset_from_index") is None
            else int(episode["dataset_from_index"])
        )
        dataset_to = (
            None
            if episode.get("dataset_to_index") is None
            else int(episode["dataset_to_index"])
        )
        if dataset_from is not None and dataset_to is not None:
            return [
                dict(row)
                for row in rows
                if dataset_from <= int(row["index"]) < dataset_to
            ]

        episode_index = (
            None
            if episode.get("episode_index") is None
            else int(episode["episode_index"])
        )
        if episode_index is None:
            return []

        matching_rows = [
            dict(row) for row in rows if int(row["episode_index"]) == episode_index
        ]
        matching_rows.sort(key=lambda row: int(row["index"]))
        return matching_rows

    def _load_json(self, relative_path: str) -> Any | None:
        path = posixpath.join(self.root_path.rstrip("/"), relative_path)
        if not self.fs.exists(path):
            return None
        with self.fs.open(path, mode="rb") as handle:
            return json.loads(handle.read())


def _format_chunk_key(chunk_key: str) -> str:
    return f"{int(chunk_key):03d}" if chunk_key.isdigit() else chunk_key


__all__ = [
    "LeRobotEpisodeReader",
    "LEROBOT_RAW_EPISODE_KEY",
    "LEROBOT_CONTEXT_KEY",
]
