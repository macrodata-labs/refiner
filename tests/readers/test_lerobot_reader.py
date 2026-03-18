from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from refiner.pipeline.sources.readers.lerobot import (
    LEROBOT_EPISODE_STATS,
    LEROBOT_TASKS,
)
from refiner.media import VideoFile
from refiner.pipeline.sources.readers.lerobot import LeRobotEpisodeReader


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


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
    shards = reader.list_shards()
    assert len(shards) == 1

    rows = list(reader.read_shard(shards[0]))
    assert len(rows) == 2
    first, second = rows

    assert int(first["episode_index"]) == 0
    assert int(second["episode_index"]) == 1
    assert first["metadata"][LEROBOT_TASKS] == {0: "pick", 1: "place"}
    assert first["metadata"][LEROBOT_TASKS] is second["metadata"][LEROBOT_TASKS]
    assert first["metadata"]["lerobot_info"] is second["metadata"]["lerobot_info"]
    assert first["metadata"]["lerobot_stats"]["observation.state"]["count"] == 2
    assert first["metadata"][LEROBOT_EPISODE_STATS]["observation.state"]["min"] == [
        -999.0
    ]
    assert "tasks" not in first
    assert "stats/observation.state/min" not in first
    assert "videos/observation.images.main/chunk_index" not in first
    assert "meta/episodes/chunk_index" not in first

    first_frames = first["frames"]
    assert isinstance(first_frames, list)
    assert len(first_frames) == 2
    assert int(first_frames[0]["frame_index"]) == 0
    assert int(first_frames[1]["frame_index"]) == 1

    video = first["observation.images.main"]
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


def test_lerobot_reader_raises_when_task_index_cannot_be_remapped() -> None:
    reader = object.__new__(LeRobotEpisodeReader)
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
        reader._remap_task_index_column(table, {0: 3, 1: 4})


def test_lerobot_reader_raises_when_frame_table_missing_task_index(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)
    _write_parquet(
        root / "data" / "chunk-000" / "file-000.parquet",
        [
            {
                "index": 0,
                "episode_index": 0,
                "frame_index": 0,
                "timestamp": 0.0,
            },
            {
                "index": 1,
                "episode_index": 0,
                "frame_index": 1,
                "timestamp": 0.1,
            },
        ],
    )

    reader = LeRobotEpisodeReader(str(root))

    with pytest.raises(
        ValueError,
        match="missing required 'task_index' column",
    ):
        list(reader.read_shard(reader.list_shards()[0]))


def test_lerobot_reader_raises_when_tasks_metadata_missing(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)
    (root / "meta" / "tasks.parquet").unlink()

    with pytest.raises(
        ValueError,
        match="missing required 'meta/tasks.parquet' metadata",
    ):
        reader = LeRobotEpisodeReader(str(root))
        list(reader.read_shard(reader.list_shards()[0]))


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
    rows = [row for shard in reader.list_shards() for row in reader.read_shard(shard)]

    assert len(rows) == 4
    assert [int(row["episode_index"]) for row in rows] == [0, 1, 2, 3]
    assert [str(row["metadata"]["lerobot_info"]["root"]) for row in rows] == [
        str(first_root),
        str(first_root),
        str(second_root),
        str(second_root),
    ]
    expected_tasks = {0: "pick", 1: "place", 2: "stack"}
    assert rows[0]["metadata"][LEROBOT_TASKS] == expected_tasks
    assert rows[2]["metadata"][LEROBOT_TASKS] == expected_tasks
    assert [int(frame["task_index"]) for frame in rows[2]["frames"]] == [1, 1]
    assert [int(frame["task_index"]) for frame in rows[3]["frames"]] == [2, 2]
