from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import describe_builtin


def estimate_camera_megasam(
    *,
    video_column: str = "file_path",
    command: Sequence[str] | str | None = None,
    result_filename: str = "megasam_trajectory.json",
    output_root: str | os.PathLike[str] | None = None,
    result_column: str = "megasam_camera",
    output_dir_column: str = "megasam/output_dir",
    timeout_s: float | None = None,
    env: Mapping[str, str] | None = None,
) -> Any:
    """Return a row mapper that runs an external MegaSAM trajectory command.

    The command must write a normalized camera trajectory JSON file at
    ``{result_path}``. Command arguments may use ``{video_path}``,
    ``{output_dir}``, and ``{result_path}`` placeholders.
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
        "robotics.egocentric:estimate_camera_megasam",
        video_column=video_column,
        result_filename=result_filename,
        result_column=result_column,
        output_dir_column=output_dir_column,
        timeout_s=timeout_s,
    )
    def _run(row: Row) -> Row:
        video_path = str(row[video_column])
        output_dir = _output_dir(
            video_path=video_path,
            output_root=output_root,
        )
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
                "MegaSAM command failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )
        trajectory = load_megasam_trajectory_file(result_path)
        return row.update(
            {
                result_column: trajectory,
                output_dir_column: str(output_dir),
            }
        )

    return _run


def load_megasam_trajectory(
    *,
    path_column: str = "megasam_trajectory_path",
    result_column: str = "megasam_camera",
) -> Any:
    """Return a row mapper that loads a normalized MegaSAM trajectory JSON."""

    if not path_column:
        raise ValueError("path_column cannot be empty")
    if not result_column:
        raise ValueError("result_column cannot be empty")

    @describe_builtin(
        "robotics.egocentric:load_megasam_trajectory",
        path_column=path_column,
        result_column=result_column,
    )
    def _load(row: Row) -> Row:
        trajectory = load_megasam_trajectory_file(Path(str(row[path_column])))
        return row.update({result_column: trajectory})

    return _load


def load_megasam_trajectory_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    trajectory_path = Path(path)
    suffix = trajectory_path.suffix.lower()
    if suffix == ".json":
        with trajectory_path.open("r", encoding="utf-8") as raw:
            payload = json.load(raw)
        if not isinstance(payload, dict):
            raise ValueError("MegaSAM trajectory JSON must contain an object")
        _validate_camera_payload(payload)
        return payload
    if suffix == ".npz":
        return camera_payload_from_megasam_npz(trajectory_path)
    raise ValueError(f"unsupported MegaSAM trajectory artifact: {trajectory_path}")


def camera_payload_from_megasam_npz(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Convert MegaSAM's ``outputs/<scene>_droid.npz`` to Refiner camera JSON."""

    data = np.load(Path(path), allow_pickle=False)
    if "cam_c2w" not in data:
        raise ValueError("MegaSAM npz must contain cam_c2w")
    transforms = np.asarray(data["cam_c2w"], dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("MegaSAM cam_c2w must be shaped Tx4x4")

    payload: dict[str, Any] = {
        "T_world_camera": transforms.tolist(),
        "source": "megasam",
    }
    if "intrinsic" in data:
        intrinsic = np.asarray(data["intrinsic"], dtype=np.float64)
        if intrinsic.shape != (3, 3):
            raise ValueError("MegaSAM intrinsic must be shaped 3x3")
        payload["intrinsic"] = intrinsic.tolist()
    if "depths" in data:
        payload["depth_artifact"] = str(Path(path))
    _validate_camera_payload(payload)
    return payload


def write_megasam_trajectory_json(
    *,
    npz_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> Path:
    payload = camera_payload_from_megasam_npz(npz_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def _validate_camera_payload(payload: dict[str, Any]) -> None:
    if "T_world_camera" not in payload:
        raise ValueError("MegaSAM camera payload requires T_world_camera")
    transforms = np.asarray(payload["T_world_camera"], dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("MegaSAM T_world_camera must be shaped Tx4x4")


def _output_dir(
    *,
    video_path: str,
    output_root: str | os.PathLike[str] | None,
) -> Path:
    digest = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:16]
    name = f"megasam-{digest}"
    if output_root is None:
        return Path(tempfile.gettempdir()) / "refiner-megasam" / name
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
        raw_command = os.environ.get("REFINER_MEGASAM_COMMAND")
    if raw_command is None:
        raise ValueError(
            "MegaSAM command is required. Pass command=... or set "
            "REFINER_MEGASAM_COMMAND."
        )
    parts = shlex.split(raw_command) if isinstance(raw_command, str) else raw_command
    values = {
        "video_path": video_path,
        "output_dir": output_dir,
        "result_path": result_path,
    }
    return [str(part).format(**values) for part in parts]


__all__ = [
    "camera_payload_from_megasam_npz",
    "estimate_camera_megasam",
    "load_megasam_trajectory",
    "load_megasam_trajectory_file",
    "write_megasam_trajectory_json",
]
