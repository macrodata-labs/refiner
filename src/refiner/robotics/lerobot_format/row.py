from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from refiner.io import DataFolder
from refiner.media import VideoFile
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.tabular import Tabular
from refiner.robotics.lerobot_format.metadata.info import DEFAULT_VIDEO_PATH
from refiner.robotics.lerobot_format.metadata.metadata import LeRobotMetadata
from refiner.robotics.lerobot_format.metadata.stats import (
    LeRobotFeatureStats,
)

LEROBOT_TASKS = "lerobot_tasks"

if TYPE_CHECKING:
    from refiner.robotics.lerobot_format.tabular import LeRobotTabular


@dataclass(frozen=True, slots=True)
class LeRobotVideoRef:
    _row: "LeRobotRow"
    key: str

    @property
    def video(self) -> VideoFile:
        return self._row._build_video_file(self.key)

    @property
    def uri(self) -> str:
        return self.video.uri

    @property
    def from_timestamp_s(self) -> float | None:
        return self.video.from_timestamp_s

    @property
    def to_timestamp_s(self) -> float | None:
        return self.video.to_timestamp_s

    def with_timestamps(
        self,
        *,
        from_timestamp_s: float | None = None,
        to_timestamp_s: float | None = None,
    ) -> "LeRobotRow":
        return self._row.with_video(
            self.key,
            from_timestamp_s=from_timestamp_s,
            to_timestamp_s=to_timestamp_s,
        )

    def with_uri(self, uri: str) -> "LeRobotRow":
        return self._row.with_video(self.key, uri=uri)

    def shift(self, delta_s: float) -> "LeRobotRow":
        from_timestamp = self._row._row.get(f"videos/{self.key}/from_timestamp")
        to_timestamp = self._row._row.get(f"videos/{self.key}/to_timestamp")
        return self._row.with_video(
            self.key,
            from_timestamp_s=(
                float(from_timestamp) + delta_s if from_timestamp is not None else None
            ),
            to_timestamp_s=(
                float(to_timestamp) + delta_s if to_timestamp is not None else None
            ),
        )


@dataclass(frozen=True, slots=True)
class LeRobotStatsView(Mapping[str, LeRobotFeatureStats]):
    _row: Row

    def __getitem__(self, feature: str) -> LeRobotFeatureStats:
        prefix = f"stats/{feature}/"
        return LeRobotFeatureStats(
            min=self._row.get(f"{prefix}min"),
            max=self._row.get(f"{prefix}max"),
            mean=self._row.get(f"{prefix}mean"),
            std=self._row.get(f"{prefix}std"),
            count=self._row.get(f"{prefix}count"),
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


@dataclass(frozen=True, slots=True)
class LeRobotVideosView(Mapping[str, LeRobotVideoRef]):
    _row: "LeRobotRow"

    def __getitem__(self, key: str) -> LeRobotVideoRef:
        row = self._row
        uri_key = f"videos/{key}/uri"
        if uri_key not in row._row:
            row = row.update({uri_key: row._build_default_video_uri(key)})
        row._build_video_file(key)
        return LeRobotVideoRef(row, key)

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
class LeRobotRow(Row):
    _row: Row
    metadata: LeRobotMetadata
    frames: Sequence[Row] | Tabular
    root: DataFolder | None = None

    def __getitem__(self, key: str) -> Any:
        if key == "metadata":
            return self.metadata
        if key == LEROBOT_TASKS:
            return self.task_index_to_text
        if key == "frames":
            return self.frames
        return self._row[key]

    def __iter__(self) -> Iterator[str]:
        fixed_keys = ("metadata", LEROBOT_TASKS, "frames")
        yield from fixed_keys
        for key in self._row:
            if key not in fixed_keys:
                yield key

    def __len__(self) -> int:
        fixed_keys = ("metadata", LEROBOT_TASKS, "frames")
        return len(self._row) + sum(1 for key in fixed_keys if key not in self._row)

    @property
    def shard_id(self) -> str | None:
        return self._row.shard_id

    @property
    def episode_index(self) -> int:
        return int(self._row["episode_index"])

    @property
    def tasks(self) -> list[str]:
        return list(self._row.get("tasks", []))

    @property
    def length(self) -> int:
        return int(self._row["length"])

    @property
    def task_index_to_text(self) -> Mapping[int, str]:
        return self.metadata.tasks.index_to_task

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
        return VideoFile(
            uri=str(uri),
            from_timestamp_s=float(from_timestamp)
            if from_timestamp is not None
            else 0.0,
            to_timestamp_s=float(to_timestamp),
        )

    def _build_default_video_uri(self, key: str) -> str:
        chunk = self._row.get(f"videos/{key}/chunk_index")
        file_idx = self._row.get(f"videos/{key}/file_index")
        root = self.root
        if chunk is None or file_idx is None or root is None:
            raise KeyError(key)
        return str(
            root.abs_paths(
                DEFAULT_VIDEO_PATH.format(
                    video_key=key,
                    chunk_index=chunk,
                    file_index=file_idx,
                )
            )
        )

    @property
    def videos(self) -> LeRobotVideosView:
        return LeRobotVideosView(self)

    @property
    def stats(self) -> LeRobotStatsView:
        return LeRobotStatsView(self._row)

    def with_video(
        self,
        key: str,
        *,
        uri: str | None = None,
        from_timestamp_s: float | None = None,
        to_timestamp_s: float | None = None,
    ) -> "LeRobotRow":
        patch: dict[str, Any] = {}
        if uri is not None:
            patch[f"videos/{key}/uri"] = uri
        if from_timestamp_s is not None:
            patch[f"videos/{key}/from_timestamp"] = from_timestamp_s
        if to_timestamp_s is not None:
            patch[f"videos/{key}/to_timestamp"] = to_timestamp_s
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
    "LEROBOT_TASKS",
    "LeRobotRow",
    "LeRobotStatsView",
    "LeRobotVideoRef",
    "LeRobotVideosView",
]
