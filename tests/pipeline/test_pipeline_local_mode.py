from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np

from refiner import hydrate_media
from refiner.execution.asyncio.runtime import get_async_runtime, submit
from refiner.pipeline import RefinerPipeline
from refiner.pipeline.data.row import DictRow, Row
from refiner.pipeline.data.shard import Shard
from refiner.pipeline.sources.readers.base import BaseReader
from refiner.media import MediaFile, Video


class _LocalFakeReader(BaseReader):
    def __init__(self, shards: list[Shard], rows_by_shard_id: dict[str, list[Row]]):
        self._shards = shards
        self._rows_by_shard_id = rows_by_shard_id

    def list_shards(self) -> list[Shard]:
        return list(self._shards)

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        yield from self._rows_by_shard_id.get(shard.id, [])


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


def test_iter_rows_is_lazy_and_crosses_shards() -> None:
    s1 = Shard(path="a", start=0, end=1)
    s2 = Shard(path="b", start=0, end=1)
    rows = {
        s1.id: [DictRow({"x": 1}), DictRow({"x": 2})],
        s2.id: [DictRow({"x": 3}), DictRow({"x": 4})],
    }
    seen_batches: list[list[int]] = []

    def batch_fn(batch: list[Row]):
        seen_batches.append([int(r["x"]) for r in batch])
        return batch

    pipeline = (
        RefinerPipeline(source=_LocalFakeReader([s1, s2], rows))
        .map(lambda r: {"x": r["x"]})
        .batch_map(batch_fn, batch_size=3)
    )

    it = iter(pipeline)
    first = next(it)
    assert first["x"] == 1
    # The first batch should include rows from both shards due to local global stream.
    assert seen_batches[0] == [1, 2, 3]


def test_materialize_and_take() -> None:
    s1 = Shard(path="a", start=0, end=1)
    s2 = Shard(path="b", start=0, end=1)
    rows = {
        s1.id: [DictRow({"x": 1}), DictRow({"x": 2})],
        s2.id: [DictRow({"x": 3})],
    }
    pipeline = RefinerPipeline(source=_LocalFakeReader([s1, s2], rows)).map(
        lambda r: {"y": r["x"] * 10}
    )

    assert [r["y"] for r in pipeline.take(2)] == [10, 20]
    assert [r["y"] for r in pipeline.materialize()] == [10, 20, 30]


def test_batch_groups_split_on_increasing_batch_size() -> None:
    s = Shard(path="a", start=0, end=1)
    rows = {s.id: [DictRow({"x": i}) for i in range(10)]}
    seen_4: list[int] = []
    seen_8: list[int] = []

    def b4(batch: list[Row]):
        seen_4.append(len(batch))
        return batch

    def b8(batch: list[Row]):
        seen_8.append(len(batch))
        return batch

    pipeline = (
        RefinerPipeline(source=_LocalFakeReader([s], rows))
        .batch_map(b4, batch_size=4)
        .batch_map(b8, batch_size=8)
    )

    out = list(pipeline.iter_rows())
    assert len(out) == 10
    assert seen_4 == [4, 4, 2]
    assert seen_8 == [8, 2]


def test_downstream_batch_waits_after_upstream_drop() -> None:
    s = Shard(path="a", start=0, end=1)
    rows = {s.id: [DictRow({"x": i}) for i in range(8)]}
    seen_b2: list[int] = []

    def drop_to_one(batch: list[Row]):
        if not batch:
            return []
        return [batch[0]]

    def b2(batch: list[Row]):
        seen_b2.append(len(batch))
        return batch

    pipeline = (
        RefinerPipeline(source=_LocalFakeReader([s], rows))
        .batch_map(drop_to_one, batch_size=4)
        .batch_map(b2, batch_size=2)
    )

    out = list(pipeline.iter_rows())
    assert len(out) == 2
    # b2 should run once with a full batch collected across two upstream outputs.
    assert seen_b2 == [2]


def test_flat_map_can_expand_rows() -> None:
    s = Shard(path="a", start=0, end=1)
    rows = {s.id: [DictRow({"x": 1}), DictRow({"x": 2})]}

    pipeline = RefinerPipeline(source=_LocalFakeReader([s], rows)).flat_map(
        lambda r: [{"x": r["x"]}, {"x": r["x"] * 10}]
    )

    out = list(pipeline.iter_rows())
    assert [r["x"] for r in out] == [1, 10, 2, 20]


def test_filter_primitive_keeps_matching_rows() -> None:
    s = Shard(path="a", start=0, end=1)
    rows = {s.id: [DictRow({"x": i}) for i in range(6)]}

    pipeline = (
        RefinerPipeline(source=_LocalFakeReader([s], rows))
        .filter(lambda r: int(r["x"]) % 2 == 0)
        .map(lambda r: {"y": int(r["x"]) + 100})
    )

    out = list(pipeline.iter_rows())
    assert [r["y"] for r in out] == [100, 102, 104]


def test_sync_map_can_offload_to_shared_runtime() -> None:
    s = Shard(path="a", start=0, end=1)
    rows = {s.id: [DictRow({"x": 1}), DictRow({"x": 2}), DictRow({"x": 3})]}
    seen_loop_ids: list[int] = []

    async def plus_one(v: int) -> int:
        seen_loop_ids.append(id(asyncio.get_running_loop()))
        await asyncio.sleep(0.001)
        return v + 1

    pipeline = RefinerPipeline(source=_LocalFakeReader([s], rows)).map(
        lambda row: {"x": submit(plus_one(int(row["x"]))).result()}
    )
    out = list(pipeline.iter_rows())
    assert [int(row["x"]) for row in out] == [2, 3, 4]

    async def _loop_id() -> int:
        return id(asyncio.get_running_loop())

    runtime_loop_id = get_async_runtime().submit(_loop_id()).result(timeout=1.0)
    assert set(seen_loop_ids) == {runtime_loop_id}


def test_map_async_can_hydrate_media_rows(tmp_path: Path) -> None:
    s = Shard(path="a", start=0, end=1)
    rows: list[Row] = []
    for i in range(5):
        video_path = tmp_path / f"episode-{i}.mp4"
        _write_video(video_path)
        rows.append(
            DictRow(
                {
                    "id": i,
                    "video": Video(
                        media=MediaFile(str(video_path)),
                        video_key="cam",
                        from_timestamp_s=0.0,
                        to_timestamp_s=0.3,
                        fps=10,
                    ),
                }
            )
        )

    pipeline = RefinerPipeline(source=_LocalFakeReader([s], {s.id: rows})).map_async(
        hydrate_media("video", decode=True),
        max_in_flight=2,
    )
    out = list(pipeline.iter_rows())
    assert [int(r["id"]) for r in out] == [0, 1, 2, 3, 4]
    for row in out:
        video = row["video"]
        assert isinstance(video, Video)
        assert video.media.frame_count == 4
