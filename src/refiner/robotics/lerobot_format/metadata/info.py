from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

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


@dataclass(frozen=True, slots=True)
class LeRobotFeatureInfo:
    dtype: str
    shape: tuple[int, ...]
    names: Any = None
    fps: int | None = None
    video_info: LeRobotVideoInfo | None = None


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

    @property
    def video_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                key
                for key, feature in self.features.items()
                if feature.dtype == "video"
            )
        )


def _parse_feature_info(feature: Mapping[str, Any]) -> LeRobotFeatureInfo:
    video_info_payload = feature.get("video_info")
    return LeRobotFeatureInfo(
        dtype=str(feature.get("dtype")),
        shape=tuple(int(value) for value in feature.get("shape", [])),
        names=feature.get("names"),
        fps=int(feature["fps"]) if feature.get("fps") is not None else None,
        video_info=(
            LeRobotVideoInfo(
                codec=video_info_payload.get("video.codec"),
                pix_fmt=video_info_payload.get("video.pix_fmt"),
                is_depth_map=video_info_payload.get("video.is_depth_map"),
                fps=(
                    int(video_info_payload["video.fps"])
                    if video_info_payload.get("video.fps") is not None
                    else None
                ),
                has_audio=video_info_payload.get("has_audio"),
            )
            if isinstance(video_info_payload, Mapping)
            else None
        ),
    )


def parse_info_json(
    payload: Mapping[str, Any],
) -> LeRobotInfo:
    return LeRobotInfo(
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
            key: _parse_feature_info(feature)
            for key, feature in payload.get("features", {}).items()
            if isinstance(key, str) and isinstance(feature, Mapping)
        },
        splits={
            key: str(value)
            for key, value in payload.get("splits", {}).items()
            if isinstance(key, str) and value is not None
        },
    )


__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_VIDEO_PATH",
    "LeRobotFeatureInfo",
    "LeRobotInfo",
    "LeRobotVideoInfo",
    "parse_info_json",
]
