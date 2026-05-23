from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import describe_builtin


def estimate_depth_lingbot(
    *,
    video_column: str = "file_path",
    command: Sequence[str] | str | None = None,
    result_filename: str = "lingbot_depth.json",
    output_root: str | os.PathLike[str] | None = None,
    result_column: str = "depth",
    output_dir_column: str = "depth/output_dir",
    timeout_s: float | None = None,
    env: Mapping[str, str] | None = None,
) -> Any:
    """Return a row mapper that runs an external LingBot-Depth command.

    The command must write a normalized JSON artifact at ``{result_path}``.
    Command arguments may use ``{video_path}``, ``{output_dir}``, and
    ``{result_path}`` placeholders.
    """

    if not video_column:
        raise ValueError("video_column cannot be empty")
    if not result_filename or Path(result_filename).name != result_filename:
        raise ValueError("result_filename must be a plain file name")
    if not result_column:
        raise ValueError("result_column cannot be empty")
    if not output_dir_column:
        raise ValueError("output_dir_column cannot be empty")
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("timeout_s must be > 0 when provided")

    @describe_builtin(
        "robotics.egocentric:estimate_depth_lingbot",
        video_column=video_column,
        result_filename=result_filename,
        result_column=result_column,
        output_dir_column=output_dir_column,
        timeout_s=timeout_s,
    )
    def _run(row: Row) -> Row:
        video_path = str(row[video_column])
        output_dir = _output_dir(video_path=video_path, output_root=output_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / result_filename
        args = _resolve_command(
            command,
            video_path=video_path,
            output_dir=str(output_dir),
            result_path=str(result_path),
        )
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=None if env is None else {**os.environ, **dict(env)},
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "LingBot-Depth command failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )
        depth = load_depth_artifact_file(result_path)
        return row.update({result_column: depth, output_dir_column: str(output_dir)})

    return _run


def load_depth_artifact(
    *,
    path_column: str = "depth_path",
    result_column: str = "depth",
) -> Any:
    """Return a row mapper that loads a normalized depth artifact JSON."""

    if not path_column:
        raise ValueError("path_column cannot be empty")
    if not result_column:
        raise ValueError("result_column cannot be empty")

    @describe_builtin(
        "robotics.egocentric:load_depth_artifact",
        path_column=path_column,
        result_column=result_column,
    )
    def _load(row: Row) -> Row:
        return row.update({result_column: load_depth_artifact_file(row[path_column])})

    return _load


def load_depth_artifact_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    artifact_path = Path(path)
    with artifact_path.open("r", encoding="utf-8") as raw:
        payload = json.load(raw)
    if not isinstance(payload, dict):
        raise ValueError("depth artifact JSON must contain an object")
    validate_depth_payload(payload)
    return payload


def validate_depth_payload(payload: dict[str, Any]) -> None:
    if "source" not in payload:
        raise ValueError("depth payload requires source")
    if "timestamps" not in payload:
        raise ValueError("depth payload requires timestamps")
    timestamps = payload["timestamps"]
    if not isinstance(timestamps, list) or not timestamps:
        raise ValueError("depth timestamps must be a non-empty list")
    if "metric_depth" in payload:
        metric = payload["metric_depth"]
        if not isinstance(metric, dict):
            raise ValueError("metric_depth must be an object when present")
        if "format" not in metric:
            raise ValueError("metric_depth requires format")
        if "frame_count" in metric and int(metric["frame_count"]) != len(timestamps):
            raise ValueError("metric_depth.frame_count must match timestamps")


def _output_dir(
    *,
    video_path: str,
    output_root: str | os.PathLike[str] | None,
) -> Path:
    digest = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:16]
    name = f"lingbot-depth-{digest}"
    if output_root is None:
        return Path(tempfile.gettempdir()) / "refiner-lingbot-depth" / name
    return Path(output_root) / name


def _resolve_command(
    command: Sequence[str] | str | None,
    *,
    video_path: str,
    output_dir: str,
    result_path: str,
) -> list[str]:
    import shlex

    raw_command = command
    if raw_command is None:
        raw_command = os.environ.get("REFINER_LINGBOT_DEPTH_COMMAND")
    if raw_command is None:
        raise ValueError(
            "LingBot-Depth command is required. Pass command=... or set "
            "REFINER_LINGBOT_DEPTH_COMMAND."
        )
    parts = shlex.split(raw_command) if isinstance(raw_command, str) else raw_command
    values = {
        "video_path": video_path,
        "output_dir": output_dir,
        "result_path": result_path,
    }
    return [str(part).format(**values) for part in parts]


__all__ = [
    "estimate_depth_lingbot",
    "load_depth_artifact",
    "load_depth_artifact_file",
    "validate_depth_payload",
]
