from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)


@dataclass(frozen=True, slots=True)
class LeRobotVideoInfo:
    codec: str | None = None
    pix_fmt: str | None = None
    is_depth_map: bool | None = None
    fps: int | None = None
    has_audio: bool | None = None

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "LeRobotVideoInfo":
        return cls(
            codec=payload.get("video.codec"),
            pix_fmt=payload.get("video.pix_fmt"),
            is_depth_map=payload.get("video.is_depth_map"),
            fps=(
                int(payload["video.fps"])
                if payload.get("video.fps") is not None
                else None
            ),
            has_audio=payload.get("has_audio"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "video.codec": self.codec,
            "video.pix_fmt": self.pix_fmt,
            "video.is_depth_map": self.is_depth_map,
            "video.fps": self.fps,
            "has_audio": self.has_audio,
        }


@dataclass(frozen=True, slots=True)
class LeRobotFeatureInfo:
    dtype: str
    shape: tuple[int, ...]
    names: Any = None
    fps: int | None = None
    video_info: LeRobotVideoInfo | None = None

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "LeRobotFeatureInfo":
        video_info_payload = payload.get("video_info")
        return cls(
            dtype=str(payload.get("dtype")),
            shape=tuple(int(value) for value in payload.get("shape", [])),
            names=payload.get("names"),
            fps=int(payload["fps"]) if payload.get("fps") is not None else None,
            video_info=(
                LeRobotVideoInfo.from_json_dict(video_info_payload)
                if isinstance(video_info_payload, Mapping)
                else None
            ),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "names": self.names,
            **({"fps": self.fps} if self.fps is not None else {}),
            **(
                {"video_info": self.video_info.to_json_dict()}
                if self.video_info is not None
                else {}
            ),
        }


@dataclass(frozen=True, slots=True)
class LeRobotInfo:
    codebase_version: str | None = None
    fps: int | None = None
    robot_type: str | None = None
    total_episodes: int | None = None
    total_frames: int | None = None
    total_tasks: int | None = None
    chunks_size: int | None = None
    data_files_size_in_mb: int | None = None
    video_files_size_in_mb: int | None = None
    data_path: str = DEFAULT_DATA_PATH
    video_path: str = DEFAULT_VIDEO_PATH
    features: Mapping[str, LeRobotFeatureInfo] = field(default_factory=dict)
    splits: Mapping[str, str] = field(default_factory=dict)
    episode_ids_in_sync: bool = True

    @property
    def video_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                key
                for key, feature in self.features.items()
                if feature.dtype == "video"
            )
        )

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "LeRobotInfo":
        return cls(
            codebase_version=(
                str(payload["codebase_version"])
                if payload.get("codebase_version") is not None
                else None
            ),
            fps=int(payload["fps"]) if payload.get("fps") is not None else None,
            robot_type=payload.get("robot_type"),
            total_episodes=(
                int(payload["total_episodes"])
                if payload.get("total_episodes") is not None
                else None
            ),
            total_frames=(
                int(payload["total_frames"])
                if payload.get("total_frames") is not None
                else None
            ),
            total_tasks=(
                int(payload["total_tasks"])
                if payload.get("total_tasks") is not None
                else None
            ),
            chunks_size=(
                int(payload["chunks_size"])
                if payload.get("chunks_size") is not None
                else None
            ),
            data_files_size_in_mb=(
                int(payload["data_files_size_in_mb"])
                if payload.get("data_files_size_in_mb") is not None
                else None
            ),
            video_files_size_in_mb=(
                int(payload["video_files_size_in_mb"])
                if payload.get("video_files_size_in_mb") is not None
                else None
            ),
            data_path=str(payload.get("data_path", DEFAULT_DATA_PATH)),
            video_path=str(payload.get("video_path", DEFAULT_VIDEO_PATH)),
            features={
                key: LeRobotFeatureInfo.from_json_dict(feature)
                for key, feature in payload.get("features", {}).items()
                if isinstance(key, str) and isinstance(feature, Mapping)
            },
            splits={
                key: str(value)
                for key, value in payload.get("splits", {}).items()
                if isinstance(key, str) and value is not None
            },
            episode_ids_in_sync=bool(payload.get("episode_ids_in_sync", True)),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "codebase_version": self.codebase_version,
            "fps": self.fps,
            "robot_type": self.robot_type,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "total_tasks": self.total_tasks,
            "chunks_size": self.chunks_size,
            "data_files_size_in_mb": self.data_files_size_in_mb,
            "video_files_size_in_mb": self.video_files_size_in_mb,
            "data_path": self.data_path,
            "video_path": self.video_path,
            "features": {
                key: feature.to_json_dict() for key, feature in self.features.items()
            },
            "splits": dict(self.splits),
            "episode_ids_in_sync": self.episode_ids_in_sync,
        }


def default_feature_info_by_key() -> dict[str, LeRobotFeatureInfo]:
    return {
        "frame_index": LeRobotFeatureInfo(dtype="int64", shape=(1,), names=None),
        "episode_index": LeRobotFeatureInfo(dtype="int64", shape=(1,), names=None),
        "index": LeRobotFeatureInfo(dtype="int64", shape=(1,), names=None),
        "task_index": LeRobotFeatureInfo(dtype="int64", shape=(1,), names=None),
    }


def infer_feature_info(
    value: Any,
    *,
    fps: int | None = None,
) -> LeRobotFeatureInfo | None:
    if isinstance(value, bool):
        return LeRobotFeatureInfo(dtype="bool", shape=(1,), names=None, fps=fps)

    array = np.asarray(value)
    if array.ndim == 0:
        if array.dtype.kind in {"i", "u"}:
            return LeRobotFeatureInfo(dtype="int64", shape=(1,), names=None, fps=fps)
        if array.dtype.kind == "f":
            return LeRobotFeatureInfo(
                dtype="float64",
                shape=(1,),
                names=None,
                fps=fps,
            )
        return None

    if array.dtype.kind in {"i", "u"}:
        dtype = "int64"
    elif array.dtype.kind == "f":
        dtype = "float64"
    elif array.dtype.kind == "b":
        dtype = "bool"
    else:
        return None
    return LeRobotFeatureInfo(
        dtype=dtype,
        shape=tuple(int(size) for size in array.shape),
        names=None,
        fps=fps,
    )


__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_VIDEO_PATH",
    "default_feature_info_by_key",
    "infer_feature_info",
    "LeRobotFeatureInfo",
    "LeRobotInfo",
    "LeRobotVideoInfo",
]
