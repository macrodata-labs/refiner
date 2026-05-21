from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from refiner.io import DataFolder
from refiner.video import VideoFile, VideoSource
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.tabular import Tabular, set_or_append_column
from refiner.robotics.lerobot_format.metadata.info import DEFAULT_VIDEO_PATH
from refiner.robotics.lerobot_format.metadata.metadata import LeRobotMetadata
from refiner.robotics.lerobot_format.metadata.stats import (
    LeRobotFeatureStats,
)
from refiner.robotics.row import RoboticsRow

if TYPE_CHECKING:
    from refiner.robotics.lerobot_format.tabular import LeRobotTabular


@dataclass(frozen=True, slots=True)
class LeRobotStatsView(Mapping[str, LeRobotFeatureStats]):
    _row: "LeRobotRow"

    def __getitem__(self, feature: str) -> LeRobotFeatureStats:
        prefix = f"stats/{feature}/"
        if not any(key.startswith(prefix) for key in self._row):
            raise KeyError(feature)
        return LeRobotFeatureStats(
            min=self._row.get(f"{prefix}min"),
            max=self._row.get(f"{prefix}max"),
            mean=self._row.get(f"{prefix}mean"),
            std=self._row.get(f"{prefix}std"),
            count=self._row.get(f"{prefix}count"),
            q01=self._row.get(f"{prefix}q01"),
            q10=self._row.get(f"{prefix}q10"),
            q50=self._row.get(f"{prefix}q50"),
            q90=self._row.get(f"{prefix}q90"),
            q99=self._row.get(f"{prefix}q99"),
        )

    def __iter__(self) -> Iterator[str]:
        yield from sorted(
            {
                key[len("stats/") :].rsplit("/", 1)[0]
                for key in self._row
                if key.startswith("stats/")
            }
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def drop(self, feature: str) -> "LeRobotRow":
        prefix = f"stats/{feature}/"
        return self._row.drop(
            *[key for key in self._row._row if key.startswith(prefix)]
        )


@dataclass(frozen=True, slots=True)
class LeRobotVideosView(Mapping[str, VideoFile]):
    _row: "LeRobotRow"

    def __getitem__(self, key: str) -> VideoFile:
        row = self._row
        uri_key = f"videos/{key}/uri"
        if uri_key not in row._row:
            row = row.update({uri_key: row._build_default_video_uri(key)})
        return row._build_video_file(key)

    def __iter__(self) -> Iterator[str]:
        video_keys = {
            key[len("videos/") :].split("/", 1)[0]
            for key in self._row._row
            if key.startswith("videos/")
        }
        yield from sorted(video_keys)

    def __len__(self) -> int:
        return sum(1 for _ in self)


@dataclass(frozen=True, slots=True)
class LeRobotRow(Row, RoboticsRow):
    _row: Row
    metadata: LeRobotMetadata
    frames: Sequence[Row] | Tabular
    root: DataFolder | None = None

    def __getitem__(self, key: str) -> Any:
        if key == "metadata":
            return self.metadata
        if key == "frames":
            return self.frames
        return self._row[key]

    def __iter__(self) -> Iterator[str]:
        fixed_keys = ("metadata", "frames")
        yield from fixed_keys
        for key in self._row:
            if key not in fixed_keys:
                yield key

    def __len__(self) -> int:
        fixed_keys = ("metadata", "frames")
        return len(self._row) + sum(1 for key in fixed_keys if key not in self._row)

    @property
    def shard_id(self) -> str | None:
        return self._row.shard_id

    @property
    def episode_index(self) -> int:
        return int(self._row["episode_index"])

    @property
    def episode_id(self) -> str:
        episode_id = self._row.get("episode_id")
        return str(episode_id) if episode_id is not None else str(self.episode_index)

    @property
    def tasks(self) -> list[str]:
        return list(self._row.get("tasks", []))

    @property
    def task(self) -> str | None:
        task = self._row.get("task")
        return task if isinstance(task, str) else None

    @property
    def num_frames(self) -> int:
        return self.to_frame_table().num_rows

    def _frame_values(self, key: str) -> Any:
        return self.to_frame_table().column(key)

    @property
    def timestamps(self) -> Any:
        return self._optional_frame_values("timestamp")

    @property
    def actions(self) -> Any:
        return self._optional_frame_values("action")

    @property
    def states(self) -> Any:
        return self._optional_frame_values("observation.state")

    def observations(self, name: str | None = None) -> Any:
        table = self.to_frame_table()
        values: dict[str, Any] = {
            key[len("observation.") :]: table.column(key)
            for key in table.names
            if key.startswith("observation.")
        }
        values.update({f"videos/{key}": video for key, video in self.videos.items()})
        if name is None:
            return values
        normalized_name = (
            name[len("observation.") :] if name.startswith("observation.") else name
        )
        return values[normalized_name]

    def _optional_frame_values(self, key: str) -> Any:
        table = self.to_frame_table()
        return table.column(key) if key in table.names else None

    def with_timestamps(self, values: Any) -> "LeRobotRow":
        return self._with_frame_values("timestamp", values)

    def with_actions(self, values: Any) -> "LeRobotRow":
        return self._with_frame_values("action", values)

    def with_observation(self, key: str, values: Any) -> "LeRobotRow":
        frame_key = key if key.startswith("observation.") else f"observation.{key}"
        return self._with_frame_values(frame_key, values)

    def _with_frame_values(self, key: str, values: Any) -> "LeRobotRow":
        frames = self.to_frame_table()
        value_type = (
            frames.table.schema.field(key).type
            if key in frames.table.column_names
            else None
        )
        column = (
            values
            if isinstance(values, (pa.Array, pa.ChunkedArray))
            else pa.array(values, type=value_type)
        )
        return self.update(
            frames=frames.with_table(set_or_append_column(frames.table, key, column))
        )

    def to_frame_table(self) -> Tabular:
        return (
            self.frames
            if isinstance(self.frames, Tabular)
            else Tabular.from_rows(self.frames)
        )

    @property
    def fps(self) -> float | None:
        return self.metadata.info.fps

    @property
    def robot_type(self) -> str | None:
        return self.metadata.info.robot_type

    @property
    def length(self) -> int:
        return int(self._row["length"])

    @property
    def tabular_type(self) -> type["LeRobotTabular"]:
        from refiner.robotics.lerobot_format.tabular import LeRobotTabular

        return LeRobotTabular

    def _build_video_file(self, key: str) -> VideoFile:
        uri = self._row.get(f"videos/{key}/uri")
        if uri is None:
            uri = self._build_default_video_uri(key)
        to_timestamp = self._row.get(f"videos/{key}/to_timestamp")
        if to_timestamp is None:
            raise KeyError(key)
        from_timestamp = self._row.get(f"videos/{key}/from_timestamp")
        if self.root is None:
            raise KeyError(key)
        return VideoFile(
            data_file=self.root.file(str(uri)),
            from_timestamp_s=float(from_timestamp)
            if from_timestamp is not None
            else 0.0,
            to_timestamp_s=float(to_timestamp),
        )

    def _build_default_video_uri(self, key: str) -> str:
        chunk = self._row.get(f"videos/{key}/chunk_index")
        file_idx = self._row.get(f"videos/{key}/file_index")
        if chunk is None or file_idx is None or self.root is None:
            raise KeyError(key)
        video_path = self.metadata.info.video_path or DEFAULT_VIDEO_PATH
        return video_path.format(
            video_key=key,
            chunk_index=chunk,
            file_index=file_idx,
        )

    @property
    def videos(self) -> LeRobotVideosView:
        return LeRobotVideosView(self)

    @property
    def stats(self) -> LeRobotStatsView:
        return LeRobotStatsView(self)

    def select_frames(self, indices: Sequence[int]) -> "LeRobotRow":
        frames = self.to_frame_table()
        selected = frames.table.take(pa.array(indices, type=pa.int64()))
        if "frame_index" in selected.column_names:
            selected = set_or_append_column(
                selected,
                "frame_index",
                pa.array(range(selected.num_rows), type=pa.int64()),
            )
        return self.update(
            {
                "length": selected.num_rows,
                "frames": frames.with_table(selected),
            }
        )

    def drop_stats(self, feature: str) -> "LeRobotRow":
        return self.stats.drop(feature)

    def with_video(
        self,
        key: str,
        video: VideoSource,
    ) -> "LeRobotRow":
        if not isinstance(video, VideoFile):
            raise TypeError("LeRobotRow.with_video requires a VideoFile")
        patch: dict[str, Any] = {}
        patch[f"videos/{key}/uri"] = video.uri
        if video.from_timestamp_s is not None:
            patch[f"videos/{key}/from_timestamp"] = video.from_timestamp_s
        if video.to_timestamp_s is not None:
            patch[f"videos/{key}/to_timestamp"] = video.to_timestamp_s
        return self.update(patch)

    def with_stats(
        self,
        feature: str,
        stats: LeRobotFeatureStats,
    ) -> "LeRobotRow":
        prefix = f"stats/{feature}/"
        return self.update(
            {
                f"{prefix}min": stats.min,
                f"{prefix}max": stats.max,
                f"{prefix}mean": stats.mean,
                f"{prefix}std": stats.std,
                f"{prefix}count": stats.count,
            }
        )

    def update(
        self, patch: Mapping[str, Any] | None = None, /, **kwargs: Any
    ) -> "LeRobotRow":
        merged = dict(patch or {})
        merged.update(kwargs)
        metadata = merged.pop("metadata", self.metadata)
        frames = merged.pop("frames", self.frames)
        return LeRobotRow(
            self._row.update(merged),
            metadata=metadata,
            frames=frames,
            root=self.root,
        )

    def drop(self, *keys: str) -> "LeRobotRow":
        return LeRobotRow(
            self._row.drop(*keys),
            metadata=self.metadata,
            frames=self.frames,
            root=self.root,
        )

    def with_shard_id(self, shard_id: str) -> "LeRobotRow":
        return LeRobotRow(
            self._row.with_shard_id(shard_id),
            metadata=self.metadata,
            frames=self.frames,
            root=self.root,
        )


__all__ = [
    "LeRobotRow",
    "LeRobotStatsView",
    "LeRobotVideosView",
]
