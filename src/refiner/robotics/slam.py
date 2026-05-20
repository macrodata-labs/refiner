from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import pyarrow as pa

from refiner.pipeline.data.row import Row
from refiner.pipeline.data.tabular import Tabular, set_or_append_column
from refiner.pipeline.planning import describe_builtin
from refiner.robotics.lerobot_format import LeRobotRow
from refiner.video import VideoFile

IntrinsicsMode = Literal["auto"]
SlamStatus = Literal["ok", "partial", "failed"]


@dataclass(frozen=True, slots=True)
class SlamEpisodeInput:
    """Input passed to a SLAM backend for one LeRobot episode."""

    row: LeRobotRow
    video_key: str
    video: VideoFile
    frames: Tabular


@dataclass(frozen=True, slots=True)
class SlamResult:
    """Backend output for one episode.

    `trajectory` must contain one row per episode frame. Columns named
    `frame_index` or `timestamp` are used only for alignment/debugging and are
    not copied into the output namespace.
    """

    trajectory: Tabular | pa.Table
    status: SlamStatus = "ok"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    scale_status: str = "unknown"
    confidence: float | None = None
    artifact_paths: Mapping[str, str] = field(default_factory=dict)


class SlamBackend(Protocol):
    @property
    def name(self) -> str: ...

    def run_batch(
        self,
        episodes: Sequence[SlamEpisodeInput],
        *,
        output_dirs: Sequence[Path],
    ) -> Iterable[SlamResult]: ...


@dataclass(frozen=True, slots=True)
class MASt3RSLAM:
    intrinsics: IntrinsicsMode | Mapping[str, float | int | None] = "auto"
    max_resolution: int | None = 720
    dense_geometry: bool = False

    @property
    def name(self) -> str:
        return "mast3r"

    def __post_init__(self) -> None:
        if self.intrinsics != "auto" and not isinstance(self.intrinsics, Mapping):
            raise TypeError("intrinsics must be 'auto' or a mapping")
        if self.max_resolution is not None and self.max_resolution <= 0:
            raise ValueError("max_resolution must be > 0 when provided")

    def run_batch(
        self,
        episodes: Sequence[SlamEpisodeInput],
        *,
        output_dirs: Sequence[Path],
    ) -> Iterable[SlamResult]:
        del episodes, output_dirs
        raise NotImplementedError(
            "MASt3RSLAM backend configuration is available, but the external "
            "MASt3R-SLAM runner is not wired in this package yet. Pass a custom "
            "SlamBackend to annotate_slam(...) to execute SLAM today."
        )


def annotate_slam(
    *,
    video_key: str,
    backend: SlamBackend | None = None,
    output_key: str | None = None,
    work_dir: str | Path | None = None,
) -> Callable[[list[Row]], Iterable[Row]]:
    """Return a batch mapper that annotates LeRobot episodes with SLAM output."""

    if not video_key:
        raise ValueError("video_key cannot be empty")
    resolved_backend = backend or MASt3RSLAM()
    backend_name = resolved_backend.name
    resolved_output_key = output_key or f"slam.{backend_name}"
    if not resolved_output_key:
        raise ValueError("output_key cannot be empty")

    @describe_builtin(
        "robotics:annotate_slam",
        video_key=video_key,
        backend=backend_name,
        output_key=resolved_output_key,
    )
    def _annotate(batch: list[Row]) -> Iterable[Row]:
        ready: list[tuple[LeRobotRow, Tabular, SlamEpisodeInput, Path]] = []
        for row in batch:
            prepared = _prepare_episode(
                row,
                video_key=video_key,
                backend_name=backend_name,
                output_key=resolved_output_key,
                work_dir=work_dir,
            )
            if isinstance(prepared, Row):
                yield prepared
            else:
                ready.append(prepared)

        if not ready:
            return

        results = list(
            resolved_backend.run_batch(
                [episode for _, _, episode, _ in ready],
                output_dirs=[output_dir for _, _, _, output_dir in ready],
            )
        )
        if len(results) != len(ready):
            raise ValueError(
                "SLAM backend must return one result per input episode: "
                f"{len(results)} != {len(ready)}"
            )

        for (row, frames, _episode, _output_dir), result in zip(
            ready,
            results,
            strict=True,
        ):
            yield _apply_slam_result(
                row,
                frames=frames,
                result=result,
                backend_name=backend_name,
                output_key=resolved_output_key,
            )

    return _annotate


def _prepare_episode(
    row: Row,
    *,
    video_key: str,
    backend_name: str,
    output_key: str,
    work_dir: str | Path | None,
) -> tuple[LeRobotRow, Tabular, SlamEpisodeInput, Path] | Row:
    if not isinstance(row, LeRobotRow):
        raise ValueError("annotate_slam requires LeRobotRow inputs")

    frames = (
        row.frames if isinstance(row.frames, Tabular) else Tabular.from_rows(row.frames)
    )
    if frames.num_rows <= 0:
        return row.update(
            {
                f"{output_key}.backend": backend_name,
                f"{output_key}.status": "failed",
                f"{output_key}.scale_status": "unknown",
                f"{output_key}.diagnostics": {"error": "episode has no frames"},
            }
        )

    video = row.videos[video_key].video
    episode = SlamEpisodeInput(
        row=row,
        video_key=video_key,
        video=video,
        frames=frames,
    )
    output_dir = _episode_output_dir(work_dir, row=row, backend_name=backend_name)
    return row, frames, episode, output_dir


def _apply_slam_result(
    row: LeRobotRow,
    *,
    frames: Tabular,
    result: SlamResult,
    backend_name: str,
    output_key: str,
) -> LeRobotRow:
    annotated_frames = _append_trajectory_columns(
        frames,
        result.trajectory,
        output_key=output_key,
    )
    patch: dict[str, Any] = {
        f"{output_key}.backend": backend_name,
        f"{output_key}.status": result.status,
        f"{output_key}.scale_status": result.scale_status,
        f"{output_key}.diagnostics": dict(result.diagnostics),
    }
    if result.confidence is not None:
        patch[f"{output_key}.confidence"] = float(result.confidence)
    for artifact_name, artifact_path in result.artifact_paths.items():
        patch[f"{output_key}.{artifact_name}_path"] = artifact_path
    if row.shard_id is not None:
        row.log_throughput("slam_episodes", 1, unit="episodes")
        row.log_throughput("slam_frames", frames.num_rows, unit="frames")
        if result.confidence is not None:
            row.log_histogram("slam_confidence", result.confidence, unit="score")
    return row.update(patch, frames=annotated_frames)


def _episode_output_dir(
    work_dir: str | Path | None,
    *,
    row: LeRobotRow,
    backend_name: str,
) -> Path:
    base = Path(work_dir) if work_dir is not None else Path.cwd() / ".refiner-slam"
    episode_index = row.get("episode_index", "unknown")
    path = base / backend_name / f"episode-{episode_index}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_trajectory_columns(
    frames: Tabular,
    trajectory: Tabular | pa.Table,
    *,
    output_key: str,
) -> Tabular:
    trajectory_table = (
        trajectory.table if isinstance(trajectory, Tabular) else trajectory
    )
    if trajectory_table.num_rows != frames.num_rows:
        raise ValueError(
            "SLAM trajectory row count must match episode frame count: "
            f"{trajectory_table.num_rows} != {frames.num_rows}"
        )

    out = frames.table
    for name in trajectory_table.column_names:
        if name in {"frame_index", "timestamp"}:
            continue
        out = set_or_append_column(
            out,
            f"{output_key}.{name}",
            trajectory_table.column(name),
        )
    return frames.with_table(out)


__all__ = [
    "MASt3RSLAM",
    "SlamBackend",
    "SlamEpisodeInput",
    "SlamResult",
    "annotate_slam",
]
