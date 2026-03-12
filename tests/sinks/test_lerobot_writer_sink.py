from __future__ import annotations

import json
import os
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

import refiner as mdr
from refiner.media import hydrate_media
from refiner.runtime.planning import execution_stages_for_pipeline


def _write_video(path: Path, *, fps: int = 10, frames: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"

        for idx in range(frames):
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            image[..., 0] = idx * 10
            image[..., 1] = 255 - idx * 10
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)


def _episode(
    *,
    episode_index: int,
    task: str,
    video_path: Path,
    from_ts: float,
    to_ts: float,
    values: list[float],
) -> dict:
    frames = [
        {
            "frame_index": i,
            "timestamp": float(i) / 10.0,
            "observation.state": [v],
        }
        for i, v in enumerate(values)
    ]
    return {
        "episode_index": episode_index,
        "task": task,
        "tasks": [task],
        "frames": frames,
        "observation.images.main": mdr.Video(
            media=mdr.MediaFile(str(video_path)),
            video_key="observation.images.main",
            from_timestamp_s=from_ts,
            to_timestamp_s=to_ts,
            fps=10,
        ),
        "metadata": {
            "lerobot_info": {
                "fps": 10,
                "robot_type": "mockbot",
            }
        },
    }


def test_write_lerobot_is_deferred_and_roundtrips(tmp_path: Path) -> None:
    src_video = tmp_path / "source" / "episode.mp4"
    _write_video(src_video)

    out_root = tmp_path / "out"
    pipeline = mdr.from_items(
        [
            _episode(
                episode_index=0,
                task="pick",
                video_path=src_video,
                from_ts=0.0,
                to_ts=0.3,
                values=[0.0, 2.0],
            ),
            _episode(
                episode_index=1,
                task="place",
                video_path=src_video,
                from_ts=0.3,
                to_ts=0.6,
                values=[4.0, 6.0],
            ),
        ],
        shard_size_rows=1,
    ).write_lerobot(str(out_root), overwrite=True)
    assert not (out_root / "meta" / "info.json").exists()

    rows = pipeline.materialize()
    assert rows == []

    assert (out_root / "meta" / "info.json").exists()
    assert (out_root / "meta" / "stats.json").exists()
    assert (out_root / "meta" / "tasks.parquet").exists()
    assert (out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").exists()
    assert not any((out_root / "meta").glob("chunk-*"))

    with (out_root / "meta" / "stats.json").open("r", encoding="utf-8") as fh:
        stats = json.load(fh)
    assert "observation.state" in stats
    assert stats["observation.state"]["mean"] == [3.0]
    assert stats["observation.state"]["count"] == [4.0]
    assert "observation.images.main" in stats

    episodes = pq.read_table(
        out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    assert "stats/observation.state/mean" in episodes.schema.names

    out_rows = mdr.read_lerobot(str(out_root), decode=False).materialize()
    assert len(out_rows) == 2
    assert out_rows[0]["task"] == "pick"
    assert out_rows[1]["task"] == "place"


def test_stage1_worker_chunk_namespaces_are_disjoint(tmp_path: Path) -> None:
    src_video = tmp_path / "source" / "episode.mp4"
    _write_video(src_video)

    out_root = tmp_path / "parallel"

    pipeline_rank0 = mdr.from_items(
        [
            _episode(
                episode_index=0,
                task="pick",
                video_path=src_video,
                from_ts=0.0,
                to_ts=0.2,
                values=[1.0, 2.0],
            )
        ]
    ).write_lerobot(str(out_root), overwrite=True)

    pipeline_rank1 = mdr.from_items(
        [
            _episode(
                episode_index=1,
                task="place",
                video_path=src_video,
                from_ts=0.2,
                to_ts=0.4,
                values=[3.0, 4.0],
            )
        ]
    ).write_lerobot(str(out_root), overwrite=False)

    stage1_rank0 = execution_stages_for_pipeline(pipeline_rank0)[0].pipeline
    stage1_rank1 = execution_stages_for_pipeline(pipeline_rank1)[0].pipeline
    reducer = execution_stages_for_pipeline(pipeline_rank0)[1].pipeline

    previous_worker_rank = os.environ.get("REFINER_WORKER_RANK")
    try:
        stage1_rank0.prepare_sinks_for_launch()

        os.environ["REFINER_WORKER_RANK"] = "0"
        list(stage1_rank0._iter_single_stage_rows())

        os.environ["REFINER_WORKER_RANK"] = "1"
        list(stage1_rank1._iter_single_stage_rows())

        chunk_dirs = sorted(path.name for path in (out_root / "meta").glob("chunk-*"))
        assert "chunk-000000" in chunk_dirs
        assert "chunk-1000000" in chunk_dirs
    finally:
        if previous_worker_rank is None:
            os.environ.pop("REFINER_WORKER_RANK", None)
        else:
            os.environ["REFINER_WORKER_RANK"] = previous_worker_rank

    reducer.materialize()

    assert not any((out_root / "meta").glob("chunk-*"))
    episodes = pq.read_table(
        out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    assert episodes.num_rows == 2


def test_write_lerobot_launch_local_runs_stage1_then_stage2(tmp_path: Path) -> None:
    out_root = tmp_path / "local-launch"
    pipeline = mdr.from_items(
        [
            {
                "episode_index": 0,
                "task": "pick",
                "tasks": ["pick"],
                "frames": [
                    {"frame_index": 0, "timestamp": 0.0, "observation.state": [1.0]},
                    {"frame_index": 1, "timestamp": 0.1, "observation.state": [2.0]},
                ],
                "metadata": {"lerobot_info": {"fps": 10, "robot_type": "mockbot"}},
            },
            {
                "episode_index": 1,
                "task": "place",
                "tasks": ["place"],
                "frames": [
                    {"frame_index": 0, "timestamp": 0.0, "observation.state": [3.0]},
                    {"frame_index": 1, "timestamp": 0.1, "observation.state": [4.0]},
                ],
                "metadata": {"lerobot_info": {"fps": 10, "robot_type": "mockbot"}},
            },
        ],
        shard_size_rows=1,
    ).write_lerobot(str(out_root), overwrite=True)

    stats = pipeline.launch_local(
        name="lerobot-two-stage-local",
        num_workers=2,
        workdir=str(tmp_path / "workdir"),
    )
    assert stats.failed == 0
    assert stats.claimed >= 3

    assert (out_root / "meta" / "info.json").exists()
    assert (out_root / "meta" / "stats.json").exists()
    assert (out_root / "meta" / "tasks.parquet").exists()
    assert not any((out_root / "meta").glob("chunk-*"))


def test_write_lerobot_accepts_decoded_videos(tmp_path: Path) -> None:
    src_video = tmp_path / "source" / "episode.mp4"
    _write_video(src_video)

    out_root = tmp_path / "decoded"
    pipeline = (
        mdr.from_items(
            [
                _episode(
                    episode_index=0,
                    task="pick",
                    video_path=src_video,
                    from_ts=0.0,
                    to_ts=0.4,
                    values=[1.0, 2.0, 3.0, 4.0],
                ),
            ]
        )
        .map_async(hydrate_media("observation.images.main", decode=True))
        .write_lerobot(str(out_root), overwrite=True)
    )

    out = pipeline.materialize()
    assert out == []

    assert (out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").exists()
    rows = pq.read_table(
        out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    assert len(rows) == 1
    assert rows["videos/observation.images.main/chunk_index"][0].as_py() == 0
