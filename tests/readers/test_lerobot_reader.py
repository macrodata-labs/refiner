from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typing import cast

from refiner.media import VideoFile
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.readers.lerobot import LeRobotEpisodeReader
from refiner.robotics.lerobot_format import (
    LEROBOT_TASKS,
    LeRobotMetadata,
    LeRobotRow,
    remap_task_index_table,
)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _episode_rows(reader: LeRobotEpisodeReader) -> list[LeRobotRow]:
    blocks = [
        block
        for shard in reader.list_shards()
        for block in reader.read_shard(shard)
        if isinstance(block, Tabular)
    ]
    assert all(isinstance(block, Tabular) for block in blocks)
    rows = [cast(LeRobotRow, row) for block in blocks for row in block.to_rows()]
    assert all(isinstance(row, LeRobotRow) for row in rows)
    return rows


def _build_sample_dataset(
    root: Path,
    *,
    index_to_task: dict[int, str] | None = None,
    episode_tasks: tuple[str, str] = ("pick", "place"),
    frame_task_indices: tuple[int, int, int, int] = (0, 0, 1, 1),
) -> None:
    index_to_task = index_to_task or {0: "pick", 1: "place"}
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "features": {"observation.images.main": {"dtype": "video"}},
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {
                    "min": [-1.0],
                    "max": [1.0],
                    "mean": [0.0],
                    "std": [0.5],
                    "count": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    _write_parquet(
        root / "meta" / "tasks.parquet",
        [
            {"task": task, "task_index": task_index}
            for task_index, task in index_to_task.items()
        ],
    )
    _write_parquet(
        root / "meta" / "episodes" / "part-000.parquet",
        [
            {
                "episode_index": 0,
                "dataset_from_index": 0,
                "dataset_to_index": 2,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                "stats/observation.state/min": [-999.0],
                "tasks": [episode_tasks[0]],
                "videos/observation.images.main/chunk_index": 0,
                "videos/observation.images.main/file_index": 0,
                "videos/observation.images.main/from_timestamp": 0.0,
                "videos/observation.images.main/to_timestamp": 1.0,
            },
            {
                "episode_index": 1,
                "dataset_from_index": 2,
                "dataset_to_index": 4,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                "stats/observation.state/min": [-999.0],
                "tasks": [episode_tasks[1]],
                "videos/observation.images.main/chunk_index": 0,
                "videos/observation.images.main/file_index": 1,
                "videos/observation.images.main/from_timestamp": 1.0,
                "videos/observation.images.main/to_timestamp": 2.0,
            },
        ],
    )
    _write_parquet(
        root / "data" / "chunk-000" / "file-000.parquet",
        [
            {
                "index": 0,
                "episode_index": 0,
                "frame_index": 0,
                "timestamp": 0.0,
                "task_index": frame_task_indices[0],
            },
            {
                "index": 1,
                "episode_index": 0,
                "frame_index": 1,
                "timestamp": 0.1,
                "task_index": frame_task_indices[1],
            },
            {
                "index": 2,
                "episode_index": 1,
                "frame_index": 0,
                "timestamp": 1.0,
                "task_index": frame_task_indices[2],
            },
            {
                "index": 3,
                "episode_index": 1,
                "frame_index": 1,
                "timestamp": 1.1,
                "task_index": frame_task_indices[3],
            },
        ],
    )


def test_lerobot_reader_emits_episode_rows(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    reader = LeRobotEpisodeReader(str(root))
    rows = _episode_rows(reader)
    assert len(rows) == 2
    first, second = rows

    assert int(first["episode_index"]) == 0
    assert int(second["episode_index"]) == 1
    assert first[LEROBOT_TASKS] == {0: "pick", 1: "place"}
    assert first[LEROBOT_TASKS] is second[LEROBOT_TASKS]
    assert isinstance(first["metadata"], LeRobotMetadata)
    assert first["metadata"].info is second["metadata"].info
    assert first["metadata"].stats["observation.state"].count == 2
    with pytest.raises(FrozenInstanceError):
        first["metadata"].info = first["metadata"].info
    assert first["tasks"] == ["pick"]
    assert first["stats/observation.state/min"] == [-999.0]
    first_frames = first["frames"]
    assert isinstance(first_frames, Tabular)
    frame_rows = first_frames.to_rows()
    assert len(frame_rows) == 2
    assert int(frame_rows[0]["frame_index"]) == 0
    assert int(frame_rows[1]["frame_index"]) == 1

    video = first.videos["observation.images.main"].video
    assert isinstance(video, VideoFile)
    assert video.uri.endswith("/videos/observation.images.main/chunk-000/file-000.mp4")
    assert video.from_timestamp_s == 0.0
    assert video.to_timestamp_s == 1.0


def test_lerobot_reader_exposes_episode_shard_planning_knobs(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    reader = LeRobotEpisodeReader(str(root), num_shards=2)
    shards = reader.list_shards()

    assert len(shards) == 2


def test_lerobot_reader_describe_uses_dataset_roots(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    reader = LeRobotEpisodeReader(str(root))

    assert reader.describe() == {
        "path": str(root),
        "inputs": [str(root)],
    }


def test_remap_task_index_table_raises_when_index_cannot_be_remapped() -> None:
    table = pa.Table.from_pydict(
        {
            "task_index": [0, 7, 1],
            "frame_index": [0, 1, 2],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"missing from source task metadata: \[7\]",
    ):
        remap_task_index_table(table, {0: 3, 1: 4})


def test_lerobot_reader_offsets_episode_indices_across_multiple_roots(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "lerobot-a"
    second_root = tmp_path / "lerobot-b"
    _build_sample_dataset(first_root)
    _build_sample_dataset(
        second_root,
        index_to_task={0: "place", 1: "stack"},
        episode_tasks=("place", "stack"),
        frame_task_indices=(0, 0, 1, 1),
    )

    reader = LeRobotEpisodeReader([str(first_root), str(second_root)])
    rows = _episode_rows(reader)

    assert len(rows) == 4
    assert [int(row["episode_index"]) for row in rows] == [0, 1, 0, 1]
    expected_tasks = {0: "pick", 1: "place", 2: "stack"}
    assert rows[0][LEROBOT_TASKS] == expected_tasks
    assert rows[2][LEROBOT_TASKS] == expected_tasks
    assert [int(frame["task_index"]) for frame in rows[2]["frames"].to_rows()] == [1, 1]
    assert [int(frame["task_index"]) for frame in rows[3]["frames"].to_rows()] == [2, 2]
