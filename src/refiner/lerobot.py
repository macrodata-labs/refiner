from __future__ import annotations

from copy import deepcopy
import math
import numbers
import posixpath
import re
from collections.abc import Callable, Mapping
from typing import Any

from refiner.sources.readers.lerobot import (
    LEROBOT_CONTEXT_KEY,
    LEROBOT_RAW_EPISODE_KEY,
)
from refiner.sources.row import DictRow, Row
from refiner.video import Video


_LEROBOT_NON_EPISODE_KEYS = {
    "frames",
    "tasks",
    "task",
    "metadata",
}
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _is_public_episode_key(key: str) -> bool:
    return (
        not key.startswith("stats/")
        and not key.startswith("videos/")
        and not key.startswith("meta/episodes/")
    )


def _build_video_from_episode(
    *,
    video_key: str,
    episode: Mapping[str, Any],
    frames: list[dict[str, Any]],
    context: Mapping[str, Any],
) -> Video | None:
    video_path_tmpl = context.get("video_path_template")
    if not isinstance(video_path_tmpl, str) or not video_path_tmpl:
        return None

    chunk = _to_int(episode.get(f"videos/{video_key}/chunk_index"))
    file_idx = _to_int(episode.get(f"videos/{video_key}/file_index"))
    if chunk is None or file_idx is None:
        return None

    rel = str(
        video_path_tmpl.format(
            video_key=video_key,
            chunk_index=chunk,
            file_index=file_idx,
        )
    )
    root_uri = str(context.get("root_uri") or "")
    uri = posixpath.join(root_uri.rstrip("/"), rel)
    first = frames[0] if frames else {}

    return Video(
        uri=uri,
        video_key=video_key,
        relative_path=rel,
        episode_index=_to_int(episode.get("episode_index")),
        frame_index=_to_int(first.get("frame_index")),
        timestamp_s=_to_float(first.get("timestamp")),
        from_timestamp_s=_to_float(episode.get(f"videos/{video_key}/from_timestamp")),
        to_timestamp_s=_to_float(episode.get(f"videos/{video_key}/to_timestamp")),
        chunk_index=chunk,
        file_index=file_idx,
        fps=_to_int(context.get("fps")),
        bytes=None,
        decode=bool(context.get("decode", False)),
    )


def to_lerobot_episode(row: Row) -> dict[str, Any]:
    """Convert a Refiner episode row into a LeRobot-style episode dict."""
    raw = _get_internal_metadata(row, LEROBOT_RAW_EPISODE_KEY)
    if not isinstance(raw, Mapping):
        raise ValueError(
            "Row is missing LeRobot raw metadata. Use this on rows from read_lerobot(...)."
        )

    episode = dict(deepcopy(raw))

    frames = row.get("frames")
    if isinstance(frames, list):
        episode["frames"] = deepcopy(frames)

    if "tasks" in row:
        episode["tasks"] = deepcopy(row["tasks"])
    if "task" in row:
        episode["task"] = row["task"]
    if "metadata" in row:
        metadata = deepcopy(row["metadata"])
        if isinstance(metadata, dict):
            metadata.pop("x", None)
        episode["metadata"] = metadata

    return episode


def from_lerobot_episode(base_row: Row, episode: Mapping[str, Any]) -> Row:
    """Rebuild a Refiner episode row from a LeRobot-style episode dict."""
    context_raw = _get_internal_metadata(base_row, LEROBOT_CONTEXT_KEY)
    if not isinstance(context_raw, Mapping):
        raise ValueError(
            "Row is missing LeRobot context metadata. Use this on rows from read_lerobot(...)."
        )

    context = dict(context_raw)
    out = base_row.to_dict()

    raw_episode = {
        k: deepcopy(v)
        for k, v in episode.items()
        if k not in _LEROBOT_NON_EPISODE_KEYS
    }

    prev_raw = _get_internal_metadata(base_row, LEROBOT_RAW_EPISODE_KEY)
    old_public = (
        {k for k in prev_raw.keys() if _is_public_episode_key(k)}
        if isinstance(prev_raw, Mapping)
        else set()
    )
    new_public = {k for k in raw_episode.keys() if _is_public_episode_key(k)}

    for key in old_public - new_public:
        out.pop(key, None)
    for key in new_public:
        out[key] = deepcopy(raw_episode[key])

    if "frames" in episode:
        out["frames"] = deepcopy(episode["frames"])
    frames_value = out.get("frames")
    frames = frames_value if isinstance(frames_value, list) else []

    if "tasks" in episode:
        out["tasks"] = deepcopy(episode["tasks"])
    if "task" in episode:
        out["task"] = episode["task"]
    elif isinstance(out.get("tasks"), list) and out["tasks"]:
        out["task"] = out["tasks"][0]

    if "metadata" in episode:
        out["metadata"] = deepcopy(episode["metadata"])
    _set_internal_metadata(out, LEROBOT_RAW_EPISODE_KEY, raw_episode)
    _set_internal_metadata(out, LEROBOT_CONTEXT_KEY, context)

    video_keys_raw = context.get("video_keys")
    video_keys = tuple(video_keys_raw) if isinstance(video_keys_raw, (list, tuple)) else ()
    for video_key in video_keys:
        if not isinstance(video_key, str):
            continue
        video = _build_video_from_episode(
            video_key=video_key,
            episode=raw_episode,
            frames=frames,
            context=context,
        )
        if video is None:
            out.pop(video_key, None)
        else:
            out[video_key] = video

    return DictRow(out, metadata={})


def convert_le_robot_fc(
    fn: Callable[[dict[str, Any]], Mapping[str, Any] | None]
) -> Callable[[Row], Row]:
    """Wrap a LeRobot-style episode-dict function as a Refiner map function.

    The wrapped function receives a mutable LeRobot-like episode dict and can:
    - mutate it in place and return `None`, or
    - return a mapping patch that is applied on top.
    """

    if not callable(fn):
        raise TypeError("fn must be callable")

    def _map(row: Row) -> Row:
        episode = to_lerobot_episode(row)
        result = fn(episode)
        if result is not None:
            episode.update(dict(result))
        return from_lerobot_episode(row, episode)

    return _map


def convert_lerobot_fc(
    fn: Callable[[dict[str, Any]], Mapping[str, Any] | None]
) -> Callable[[Row], Row]:
    """Alias for `convert_le_robot_fc`."""
    return convert_le_robot_fc(fn)


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        float_value = float(value)
        if math.isfinite(float_value) and float_value.is_integer():
            return int(float_value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return int(stripped) if _INT_RE.match(stripped) else None
    return None


def _to_float(value: Any) -> float | None:
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


def _get_internal_metadata(row: Row, key: str) -> Any:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        extra = metadata.get("x")
        if isinstance(extra, Mapping):
            return extra.get(key)
    return row.get(key)


def _set_internal_metadata(row: dict[str, Any], key: str, value: Any) -> None:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        out_metadata = dict(deepcopy(metadata))
    else:
        out_metadata = {}

    extra = out_metadata.get("x")
    if isinstance(extra, Mapping):
        out_extra = dict(deepcopy(extra))
    else:
        out_extra = {}

    out_extra[key] = deepcopy(value)
    out_metadata["x"] = out_extra
    row["metadata"] = out_metadata


__all__ = [
    "convert_le_robot_fc",
    "convert_lerobot_fc",
    "from_lerobot_episode",
    "to_lerobot_episode",
]
