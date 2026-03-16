from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from refiner.media import Video
from refiner.pipeline.sources.readers.lerobot import (
    LEROBOT_EPISODE_STATS,
    LeRobotEpisodeReader,
)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _build_sample_dataset(root: Path) -> None:
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
            {"task_index": 0, "task": "pick"},
            {"task_index": 1, "task": "place"},
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
                "tasks": ["pick"],
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
                "tasks": ["place"],
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
                "task_index": 0,
            },
            {
                "index": 1,
                "episode_index": 0,
                "frame_index": 1,
                "timestamp": 0.1,
                "task_index": 0,
            },
            {
                "index": 2,
                "episode_index": 1,
                "frame_index": 0,
                "timestamp": 1.0,
                "task_index": 1,
            },
            {
                "index": 3,
                "episode_index": 1,
                "frame_index": 1,
                "timestamp": 1.1,
                "task_index": 1,
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
    assert first["task"] == "pick"
    assert second["task"] == "place"
    assert first["metadata"]["lerobot_stats"]["observation.state"]["count"] == 2
    assert first["metadata"][LEROBOT_EPISODE_STATS]["observation.state"]["min"] == [
        -999.0
    ]
    assert "stats/observation.state/min" not in first
    assert "videos/observation.images.main/chunk_index" not in first
    assert "meta/episodes/chunk_index" not in first

    first_frames = first["frames"]
    assert isinstance(first_frames, list)
    assert len(first_frames) == 2
    assert int(first_frames[0]["frame_index"]) == 0
    assert int(first_frames[1]["frame_index"]) == 1

    video = first["observation.images.main"]
    assert isinstance(video, Video)
    assert video.uri.endswith("/videos/observation.images.main/chunk-000/file-000.mp4")
    assert video.from_timestamp_s == 0.0
    assert video.to_timestamp_s == 1.0


def test_lerobot_reader_exposes_episode_shard_planning_knobs(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_sample_dataset(root)

    reader = LeRobotEpisodeReader(str(root), num_shards=2)
    shards = reader.list_shards()

    assert len(shards) == 2


def test_lerobot_reader_offsets_episode_indices_across_multiple_roots(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "lerobot-a"
    second_root = tmp_path / "lerobot-b"
    _build_sample_dataset(first_root)
    _build_sample_dataset(second_root)

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
