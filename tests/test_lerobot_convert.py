from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import refiner as mdr
from refiner.sources.row import DictRow


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _build_dataset(root: Path) -> None:
    (root / "meta").mkdir(parents=True, exist_ok=True)
    info = {
        "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.main": {"dtype": "video"},
            "episode_index": {"dtype": "int64"},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    _write_parquet(
        root / "meta" / "tasks.parquet",
        [{"task_index": 0, "task": "pick"}],
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
                "videos/observation.images.main/chunk_index": 0,
                "videos/observation.images.main/file_index": 0,
                "videos/observation.images.main/from_timestamp": 0.0,
                "videos/observation.images.main/to_timestamp": 1.0,
                "stats/observation.state/min": [-1.0],
            }
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
        ],
    )


def test_convert_le_robot_fc_round_trip_updates_video_and_episode(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    _build_dataset(root)

    def _edit(ep: dict) -> dict:
        ep["episode_index"] = 42
        ep["videos/observation.images.main/file_index"] = 9
        return ep

    rows = mdr.read_lerobot(str(root)).map(mdr.convert_le_robot_fc(_edit)).materialize()
    assert len(rows) == 1

    row = rows[0]
    assert int(row["episode_index"]) == 42
    assert int(row["metadata"]["x"]["__lerobot_episode"]["episode_index"]) == 42
    assert row["observation.images.main"].file_index == 9
    assert row["observation.images.main"].uri.endswith("/file-009.mp4")

    lr = mdr.to_lerobot_episode(row)
    assert int(lr["episode_index"]) == 42
    assert int(lr["videos/observation.images.main/file_index"]) == 9
    assert "stats/observation.state/min" in lr


def test_convert_le_robot_fc_requires_lerobot_rows() -> None:
    fn = mdr.convert_le_robot_fc(lambda ep: ep)
    with pytest.raises(ValueError):
        fn(DictRow({"x": 1}, metadata={}))
