from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import describe_builtin
from refiner.robotics.egocentric.depth import validate_depth_payload

ExtrinsicConvention = Literal["world_to_camera", "camera_to_world"]


def estimate_geometry_vggt_omega(
    *,
    video_column: str = "file_path",
    command: Sequence[str] | str | None = None,
    result_filename: str = "vggt_omega_geometry.json",
    output_root: str | os.PathLike[str] | None = None,
    result_column: str = "vggt_omega",
    camera_column: str = "vggt_omega_camera",
    depth_column: str = "vggt_omega_depth",
    output_dir_column: str = "vggt_omega/output_dir",
    timeout_s: float | None = None,
    env: Mapping[str, str] | None = None,
) -> Any:
    """Return a row mapper that runs an external VGGT-Omega command.

    The command must write a normalized VGGT-Omega geometry JSON file at
    ``{result_path}``. Command arguments may use ``{video_path}``,
    ``{output_dir}``, and ``{result_path}`` placeholders.
    """

    if not video_column:
        raise ValueError("video_column cannot be empty")
    if not result_filename or Path(result_filename).name != result_filename:
        raise ValueError("result_filename must be a plain file name")
    for name, value in {
        "result_column": result_column,
        "camera_column": camera_column,
        "depth_column": depth_column,
        "output_dir_column": output_dir_column,
    }.items():
        if not value:
            raise ValueError(f"{name} cannot be empty")
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("timeout_s must be > 0 when provided")

    @describe_builtin(
        "robotics.egocentric:estimate_geometry_vggt_omega",
        video_column=video_column,
        result_filename=result_filename,
        result_column=result_column,
        camera_column=camera_column,
        depth_column=depth_column,
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
                "VGGT-Omega command failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )
        geometry = load_vggt_omega_geometry_file(result_path)
        return row.update(
            {
                result_column: geometry,
                camera_column: geometry["camera"],
                depth_column: geometry["depth"],
                output_dir_column: str(output_dir),
            }
        )

    return _run


def load_vggt_omega_geometry(
    *,
    path_column: str = "vggt_omega_geometry_path",
    result_column: str = "vggt_omega",
    camera_column: str = "vggt_omega_camera",
    depth_column: str = "vggt_omega_depth",
) -> Any:
    """Return a row mapper that loads a normalized VGGT-Omega artifact."""

    for name, value in {
        "path_column": path_column,
        "result_column": result_column,
        "camera_column": camera_column,
        "depth_column": depth_column,
    }.items():
        if not value:
            raise ValueError(f"{name} cannot be empty")

    @describe_builtin(
        "robotics.egocentric:load_vggt_omega_geometry",
        path_column=path_column,
        result_column=result_column,
        camera_column=camera_column,
        depth_column=depth_column,
    )
    def _load(row: Row) -> Row:
        geometry = load_vggt_omega_geometry_file(Path(str(row[path_column])))
        return row.update(
            {
                result_column: geometry,
                camera_column: geometry["camera"],
                depth_column: geometry["depth"],
            }
        )

    return _load


def load_vggt_omega_geometry_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    artifact_path = Path(path)
    suffix = artifact_path.suffix.lower()
    if suffix == ".json":
        with artifact_path.open("r", encoding="utf-8") as raw:
            payload = json.load(raw)
        if not isinstance(payload, dict):
            raise ValueError("VGGT-Omega geometry JSON must contain an object")
        validate_vggt_omega_geometry_payload(payload)
        return payload
    if suffix == ".npz":
        return geometry_payload_from_vggt_omega_npz(artifact_path)
    raise ValueError(f"unsupported VGGT-Omega geometry artifact: {artifact_path}")


def geometry_payload_from_vggt_omega_npz(
    path: str | os.PathLike[str],
    *,
    extrinsic_convention: ExtrinsicConvention = "world_to_camera",
) -> dict[str, Any]:
    """Convert an official VGGT-Omega-style npz artifact to Refiner JSON.

    VGGT-style APIs usually expose camera extrinsics as world-to-camera matrices.
    Refiner's camera contract uses ``T_world_camera``. Pass
    ``extrinsic_convention="camera_to_world"`` only when the stored matrices are
    already camera-to-world transforms.
    """

    artifact_path = Path(path)
    data = np.load(artifact_path, allow_pickle=False)
    transforms = _load_transforms(data, extrinsic_convention=extrinsic_convention)
    frame_count = int(transforms.shape[0])
    timestamps = _load_timestamps(data, frame_count=frame_count)

    camera: dict[str, Any] = {
        "source": "vggt-omega",
        "T_world_camera": transforms.tolist(),
    }
    intrinsics = _optional_array(data, ("intrinsics", "intrinsic", "K"))
    if intrinsics is not None:
        if intrinsics.shape == (3, 3):
            camera["intrinsic"] = intrinsics.astype(np.float64).tolist()
        elif intrinsics.shape == (frame_count, 3, 3):
            camera["intrinsics"] = intrinsics.astype(np.float64).tolist()
        else:
            raise ValueError("VGGT-Omega intrinsics must be shaped 3x3 or Tx3x3")
    image_size = _optional_array(data, ("image_size", "image_shape"))
    if image_size is not None:
        camera["image_size"] = np.asarray(image_size).astype(int).tolist()

    depth: dict[str, Any] = {
        "source": "vggt-omega",
        "timestamps": timestamps,
        "metric_depth": {
            "format": "vggt_omega_npz",
            "frame_count": frame_count,
            "artifact": str(artifact_path),
        },
    }
    depth_key = _first_present_key(data, ("depth", "depths", "depth_map", "depth_maps"))
    if depth_key is not None:
        depth["metric_depth"]["key"] = depth_key
    confidence_key = _first_present_key(
        data,
        ("depth_conf", "depth_confidence", "depth_confs"),
    )
    if confidence_key is not None:
        depth["metric_depth"]["confidence_key"] = confidence_key

    payload = {
        "source": "vggt-omega",
        "timestamps": timestamps,
        "camera": camera,
        "depth": depth,
        "artifact": str(artifact_path),
    }
    validate_vggt_omega_geometry_payload(payload)
    return payload


def write_vggt_omega_geometry_json(
    *,
    npz_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    extrinsic_convention: ExtrinsicConvention = "world_to_camera",
) -> Path:
    payload = geometry_payload_from_vggt_omega_npz(
        npz_path,
        extrinsic_convention=extrinsic_convention,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def validate_vggt_omega_geometry_payload(payload: dict[str, Any]) -> None:
    if payload.get("source") != "vggt-omega":
        raise ValueError("VGGT-Omega geometry payload source must be 'vggt-omega'")
    timestamps = payload.get("timestamps")
    if not isinstance(timestamps, list) or not timestamps:
        raise ValueError("VGGT-Omega geometry payload requires timestamps")

    camera = payload.get("camera")
    if not isinstance(camera, dict):
        raise ValueError("VGGT-Omega geometry payload requires camera object")
    transforms = np.asarray(camera.get("T_world_camera"), dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("VGGT-Omega camera.T_world_camera must be shaped Tx4x4")
    if len(transforms) != len(timestamps):
        raise ValueError(
            "VGGT-Omega camera.T_world_camera length must match timestamps"
        )

    if "intrinsic" in camera:
        intrinsic = np.asarray(camera["intrinsic"], dtype=np.float64)
        if intrinsic.shape != (3, 3):
            raise ValueError("VGGT-Omega camera.intrinsic must be shaped 3x3")
    if "intrinsics" in camera:
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
        if intrinsics.shape != (len(timestamps), 3, 3):
            raise ValueError("VGGT-Omega camera.intrinsics must be shaped Tx3x3")

    depth = payload.get("depth")
    if not isinstance(depth, dict):
        raise ValueError("VGGT-Omega geometry payload requires depth object")
    validate_depth_payload(depth)


def _load_transforms(
    data: np.lib.npyio.NpzFile,
    *,
    extrinsic_convention: ExtrinsicConvention,
) -> np.ndarray:
    if "T_world_camera" in data:
        return _as_4x4_series(np.asarray(data["T_world_camera"], dtype=np.float64))
    if "cam_c2w" in data:
        return _as_4x4_series(np.asarray(data["cam_c2w"], dtype=np.float64))

    extrinsic_key = _first_present_key(
        data,
        ("extrinsics", "extrinsic", "camera_extrinsics", "T_camera_world", "w2c"),
    )
    if extrinsic_key is None:
        raise ValueError(
            "VGGT-Omega npz must contain extrinsics or T_world_camera/cam_c2w"
        )
    extrinsics = _as_4x4_series(np.asarray(data[extrinsic_key], dtype=np.float64))
    if extrinsic_convention == "camera_to_world":
        return extrinsics
    if extrinsic_convention == "world_to_camera":
        return np.linalg.inv(extrinsics)
    raise ValueError(f"unsupported extrinsic convention: {extrinsic_convention}")


def _as_4x4_series(transforms: np.ndarray) -> np.ndarray:
    if transforms.ndim != 3:
        raise ValueError("camera transforms must be shaped Tx3x4 or Tx4x4")
    if transforms.shape[1:] == (4, 4):
        return transforms.astype(np.float64)
    if transforms.shape[1:] == (3, 4):
        bottom = np.zeros((transforms.shape[0], 1, 4), dtype=np.float64)
        bottom[:, 0, 3] = 1.0
        return np.concatenate([transforms.astype(np.float64), bottom], axis=1)
    raise ValueError("camera transforms must be shaped Tx3x4 or Tx4x4")


def _load_timestamps(
    data: np.lib.npyio.NpzFile,
    *,
    frame_count: int,
) -> list[float]:
    timestamps = _optional_array(data, ("timestamps", "timestamp", "times"))
    if timestamps is None:
        return [float(index) for index in range(frame_count)]
    values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if len(values) != frame_count:
        raise ValueError("VGGT-Omega timestamps length must match camera frames")
    return values.tolist()


def _optional_array(
    data: np.lib.npyio.NpzFile,
    keys: Sequence[str],
) -> np.ndarray | None:
    key = _first_present_key(data, keys)
    if key is None:
        return None
    return np.asarray(data[key])


def _first_present_key(
    data: np.lib.npyio.NpzFile,
    keys: Sequence[str],
) -> str | None:
    names = set(data.files)
    for key in keys:
        if key in names:
            return key
    return None


def _output_dir(
    *,
    video_path: str,
    output_root: str | os.PathLike[str] | None,
) -> Path:
    digest = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:16]
    name = f"vggt-omega-{digest}"
    if output_root is None:
        return Path(tempfile.gettempdir()) / "refiner-vggt-omega" / name
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
        raw_command = os.environ.get("REFINER_VGGT_OMEGA_COMMAND")
    if raw_command is None:
        raise ValueError(
            "VGGT-Omega command is required. Pass command=... or set "
            "REFINER_VGGT_OMEGA_COMMAND."
        )
    parts = shlex.split(raw_command) if isinstance(raw_command, str) else raw_command
    values = {
        "video_path": video_path,
        "output_dir": output_dir,
        "result_path": result_path,
    }
    return [str(part).format(**values) for part in parts]


__all__ = [
    "ExtrinsicConvention",
    "estimate_geometry_vggt_omega",
    "geometry_payload_from_vggt_omega_npz",
    "load_vggt_omega_geometry",
    "load_vggt_omega_geometry_file",
    "validate_vggt_omega_geometry_payload",
    "write_vggt_omega_geometry_json",
]
