from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType, SimpleNamespace
import sys

from refiner.pipeline import RefinerPipeline
from refiner.pipeline.data.row import DictRow, Row
from refiner.pipeline.data.shard import FilePart, Shard
from refiner.pipeline.sources.readers.base import BaseReader
from refiner.robotics.egocentric import (
    hand_tracking_flush_row,
    is_hand_tracking_flush_row,
    run_hand_tracking,
    track_hands_egovision,
)


class _Reader(BaseReader):
    def __init__(self, rows: list[Row]) -> None:
        self._shard = Shard.from_file_parts([FilePart(path="episodes", start=0, end=1)])
        self._rows = rows

    def list_shards(self) -> list[Shard]:
        return [self._shard]

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        yield from self._rows


def test_run_hand_tracking_buffers_episode_rows_until_batch_is_full() -> None:
    seen_batches: list[list[str]] = []
    models: dict[str, str] | None = None

    def annotate_batch(rows: list[Row]):
        nonlocal models
        if models is None:
            models = {"name": "loaded"}
        episode_ids = [str(row["episode_id"]) for row in rows]
        seen_batches.append(episode_ids)
        return [
            {"episode_id": episode_id, "model": models["name"]}
            for episode_id in episode_ids
        ]

    rows: list[Row] = [DictRow({"episode_id": f"e{i}"}) for i in range(5)]
    pipeline = RefinerPipeline(source=_Reader(rows)).flat_map(
        run_hand_tracking(
            annotate_batch,
            batch_size=2,
        )
    )

    output = list(pipeline.iter_rows())

    assert [row["episode_id"] for row in output] == ["e0", "e1", "e2", "e3"]
    assert [row["model"] for row in output] == ["loaded"] * 4
    assert seen_batches == [["e0", "e1"], ["e2", "e3"]]


def test_run_hand_tracking_flushes_final_partial_batch_with_sentinel() -> None:
    calls = 0

    def annotate_batch(rows: list[Row]):
        nonlocal calls
        calls += 1
        return [{"episode_id": row["episode_id"], "call": calls} for row in rows]

    rows: list[Row] = [
        DictRow({"episode_id": "e0"}),
        DictRow({"episode_id": "e1"}),
        DictRow({"episode_id": "e2"}),
        DictRow(hand_tracking_flush_row()),
    ]
    pipeline = RefinerPipeline(source=_Reader(rows)).flat_map(
        run_hand_tracking(
            annotate_batch,
            batch_size=4,
            flush_when=is_hand_tracking_flush_row,
        )
    )

    output = list(pipeline.iter_rows())

    assert [row["episode_id"] for row in output] == ["e0", "e1", "e2"]
    assert [row["call"] for row in output] == [1, 1, 1]


def test_run_hand_tracking_can_include_flush_row() -> None:
    def annotate_batch(rows: list[Row]):
        return [{"episode_ids": [row.get("episode_id", "flush") for row in rows]}]

    rows: list[Row] = [
        DictRow({"episode_id": "e0"}),
        DictRow(hand_tracking_flush_row(episode_id="flush")),
    ]
    pipeline = RefinerPipeline(source=_Reader(rows)).flat_map(
        run_hand_tracking(
            annotate_batch,
            batch_size=8,
            flush_when=is_hand_tracking_flush_row,
            include_flush_row=True,
        )
    )

    output = list(pipeline.iter_rows())

    assert [row["episode_ids"] for row in output] == [["e0", "flush"]]


def test_run_hand_tracking_rejects_invalid_batch_size() -> None:
    try:
        run_hand_tracking(lambda rows: rows, batch_size=0)
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("expected invalid batch size to fail")


def test_track_hands_egovision_runs_episode_batch_map(monkeypatch) -> None:
    calls: list[list[list[str]]] = []
    created_configs: list[object] = []

    class EpisodeInput:
        def __init__(self, *, frames, metadata):
            self.frames = frames
            self.metadata = metadata

    class HaworReconstructionConfig:
        pass

    class HandTrackingConfig:
        def __init__(self, *, hand_reconstruction):
            self.hand_reconstruction = hand_reconstruction
            created_configs.append(self)

    class HandTrackingPipeline:
        def __init__(self, config):
            self.config = config

        def predict_episodes(self, episodes):
            calls.append([list(episode.frames) for episode in episodes])
            return [
                SimpleNamespace(
                    to_dict=lambda index=index, episode=episode: {
                        "episode_id": f"episode-{index}",
                        "metadata": episode.metadata,
                        "camera_trajectory": [[[1, 0, 0, 0]]],
                    }
                )
                for index, episode in enumerate(episodes)
            ]

    fake_egovision = ModuleType("egovision")
    setattr(fake_egovision, "EpisodeInput", EpisodeInput)
    setattr(fake_egovision, "HandTrackingConfig", HandTrackingConfig)
    setattr(fake_egovision, "HandTrackingPipeline", HandTrackingPipeline)
    setattr(fake_egovision, "HaworReconstructionConfig", HaworReconstructionConfig)
    monkeypatch.setitem(sys.modules, "egovision", fake_egovision)

    rows: list[Row] = [
        DictRow({"episode_id": "e0", "frames": ["f0"], "task": "pick"}),
        DictRow({"episode_id": "e1", "frames": ["f1"], "task": "place"}),
        DictRow({"episode_id": "e2", "frames": ["f2"], "task": "open"}),
    ]
    pipeline = RefinerPipeline(source=_Reader(rows)).batch_map(
        track_hands_egovision(metadata_keys=("task",)),
        batch_size=2,
    )

    output = list(pipeline.iter_rows())

    assert len(created_configs) == 1
    assert calls == [[["f0"], ["f1"]], [["f2"]]]
    assert [row["episode_id"] for row in output] == ["e0", "e1", "e2"]
    assert [row["hand_tracking"]["episode_id"] for row in output] == [
        "episode-0",
        "episode-1",
        "episode-0",
    ]
    assert [row["hand_tracking"]["metadata"]["task"] for row in output] == [
        "pick",
        "place",
        "open",
    ]
