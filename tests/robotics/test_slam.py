from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pytest

from refiner.io import DataFolder
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.tabular import Tabular
from refiner.robotics import MASt3RSLAM, SlamEpisodeInput, SlamResult, annotate_slam
from refiner.robotics.lerobot_format import (
    LeRobotInfo,
    LeRobotMetadata,
    LeRobotRow,
    LeRobotStatsFile,
    LeRobotTasks,
)


class FakeMASt3RBackend:
    name = "mast3r"

    def __init__(self, trajectory: pa.Table):
        self.trajectory = trajectory
        self.seen_episodes: list[SlamEpisodeInput] = []
        self.seen_output_dirs: list[Path] = []

    def run_batch(
        self,
        episodes: Sequence[SlamEpisodeInput],
        *,
        output_dirs: Sequence[Path],
    ) -> list[SlamResult]:
        self.seen_episodes = list(episodes)
        self.seen_output_dirs = list(output_dirs)
        return [
            SlamResult(
                trajectory=self.trajectory,
                diagnostics={"num_resets": 0},
                scale_status="unknown",
                confidence=0.75,
                artifact_paths={"geometry": str(output_dir / "pointcloud.ply")},
            )
            for output_dir in output_dirs
        ]


def _row() -> LeRobotRow:
    frames = [
        DictRow({"frame_index": 0, "timestamp": 0.0, "task_index": 0}),
        DictRow({"frame_index": 1, "timestamp": 0.1, "task_index": 0}),
    ]
    return LeRobotRow(
        DictRow(
            {
                "episode_index": 7,
                "length": len(frames),
                "videos/observation.images.ego/chunk_index": 0,
                "videos/observation.images.ego/file_index": 0,
                "videos/observation.images.ego/from_timestamp": 0.0,
                "videos/observation.images.ego/to_timestamp": 0.2,
            }
        ),
        metadata=LeRobotMetadata(
            info=LeRobotInfo(fps=10, robot_type="mockbot"),
            stats=LeRobotStatsFile({}),
            tasks=LeRobotTasks({0: "pick"}),
        ),
        frames=frames,
        root=DataFolder.resolve("/tmp"),
    )


def test_annotate_slam_adds_mast3r_episode_and_frame_outputs(tmp_path: Path) -> None:
    backend = FakeMASt3RBackend(
        pa.table(
            {
                "frame_index": [0, 1],
                "world_T_camera": [[1.0] * 16, [2.0] * 16],
                "tracking_confidence": [0.8, 0.7],
            }
        )
    )

    annotated_rows = list(
        annotate_slam(
            video_key="observation.images.ego",
            backend=backend,
            output_key="slam.mast3r",
            work_dir=tmp_path,
        )([_row()])
    )
    annotated = annotated_rows[0]

    assert isinstance(annotated, LeRobotRow)
    assert len(annotated_rows) == 1
    assert len(backend.seen_episodes) == 1
    assert backend.seen_episodes[0].video_key == "observation.images.ego"
    assert backend.seen_episodes[0].frames.num_rows == 2
    assert backend.seen_output_dirs == [tmp_path / "mast3r" / "episode-7"]
    assert annotated["slam.mast3r.backend"] == "mast3r"
    assert annotated["slam.mast3r.status"] == "ok"
    assert annotated["slam.mast3r.scale_status"] == "unknown"
    assert annotated["slam.mast3r.confidence"] == pytest.approx(0.75)
    assert annotated["slam.mast3r.diagnostics"] == {"num_resets": 0}
    assert annotated["slam.mast3r.geometry_path"].endswith("pointcloud.ply")

    frames = annotated.frames
    assert isinstance(frames, Tabular)
    assert "slam.mast3r.world_T_camera" in frames.table.column_names
    assert "slam.mast3r.tracking_confidence" in frames.table.column_names
    assert frames.table.column("slam.mast3r.tracking_confidence").to_pylist() == [
        0.8,
        0.7,
    ]


def test_annotate_slam_rejects_mismatched_trajectory_length(tmp_path: Path) -> None:
    backend = FakeMASt3RBackend(pa.table({"world_T_camera": [[1.0] * 16]}))

    with pytest.raises(ValueError, match="row count must match"):
        list(
            annotate_slam(
                video_key="observation.images.ego",
                backend=backend,
                work_dir=tmp_path,
            )([_row()])
        )


def test_mast3r_slam_validates_resolution() -> None:
    with pytest.raises(ValueError, match="max_resolution"):
        MASt3RSLAM(max_resolution=0)
