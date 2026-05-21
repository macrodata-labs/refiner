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
from refiner.robotics.egocentric.types import HaworResult


def reconstruct_hands_hawor(
    *,
    video_column: str = "file_path",
    command: Sequence[str] | str | None = None,
    result_filename: str = "hawor_result.json",
    output_root: str | os.PathLike[str] | None = None,
    result_column: str = "hawor",
    output_dir_column: str = "hawor/output_dir",
    timeout_s: float | None = None,
    env: Mapping[str, str] | None = None,
) -> Any:
    """Return a row mapper that runs an external HaWoR reconstruction command.

    The command must write a normalized JSON result into
    ``{output_dir}/{result_filename}``. Command arguments may use these
    placeholders:

    - ``{video_path}``
    - ``{output_dir}``
    - ``{result_path}``

    If ``command`` is omitted, ``REFINER_HAWOR_COMMAND`` is used and parsed as a
    shell-style command string. Refiner intentionally does not depend on HaWoR's
    research runtime or weights.
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
        "robotics.egocentric:reconstruct_hands_hawor",
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
                "HaWoR command failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )
        result = load_hawor_result_file(result_path)
        if row.shard_id is not None:
            row.log_throughput("hawor_videos_processed", 1, unit="videos")
            row.log_throughput(
                "hawor_frames_processed",
                len(result.timestamps),
                unit="frames",
            )
        return row.update(
            {
                result_column: result.to_dict(),
                output_dir_column: str(output_dir),
            }
        )

    return _run


def load_hawor_result(
    *,
    path_column: str = "hawor_result_path",
    result_column: str = "hawor",
) -> Any:
    """Return a row mapper that loads a normalized HaWoR JSON artifact."""

    if not path_column:
        raise ValueError("path_column cannot be empty")
    if not result_column:
        raise ValueError("result_column cannot be empty")

    @describe_builtin(
        "robotics.egocentric:load_hawor_result",
        path_column=path_column,
        result_column=result_column,
    )
    def _load(row: Row) -> Row:
        result = load_hawor_result_file(Path(str(row[path_column])))
        return row.update({result_column: result.to_dict()})

    return _load


def load_hawor_result_file(path: str | os.PathLike[str]) -> HaworResult:
    result_path = Path(path)
    with result_path.open("r", encoding="utf-8") as raw:
        payload = json.load(raw)
    if not isinstance(payload, dict):
        raise ValueError("HaWoR result JSON must contain an object")
    return HaworResult.from_mapping(payload)


def _output_dir(
    *,
    video_path: str,
    output_root: str | os.PathLike[str] | None,
) -> Path:
    digest = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:16]
    name = f"hawor-{digest}"
    if output_root is None:
        return Path(tempfile.gettempdir()) / "refiner-hawor" / name
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
        raw_command = os.environ.get("REFINER_HAWOR_COMMAND")
    if raw_command is None:
        raise ValueError(
            "HaWoR command is required. Pass command=... or set REFINER_HAWOR_COMMAND."
        )
    parts = shlex.split(raw_command) if isinstance(raw_command, str) else raw_command
    values = {
        "video_path": video_path,
        "output_dir": output_dir,
        "result_path": result_path,
    }
    return [str(part).format(**values) for part in parts]


__all__ = [
    "load_hawor_result",
    "load_hawor_result_file",
    "reconstruct_hands_hawor",
]
