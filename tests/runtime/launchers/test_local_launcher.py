from __future__ import annotations

from collections.abc import Iterator

import pytest

from refiner.ledger.shard import Shard
from refiner.pipeline import RefinerPipeline, read_jsonl
from refiner.sources.readers.base import BaseReader
from refiner.sources.row import DictRow, Row
from refiner.runtime.cpu import build_cpu_sets


class _FakeReader(BaseReader):
    def __init__(self, shards: list[Shard], rows_by_shard_id: dict[str, list[Row]]):
        self._shards = shards
        self._rows_by_shard_id = rows_by_shard_id

    def list_shards(self) -> list[Shard]:
        return list(self._shards)

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        yield from self._rows_by_shard_id.get(shard.id, [])


def test_launch_local_single_worker() -> None:
    s1 = Shard(path="a", start=0, end=1)
    s2 = Shard(path="b", start=0, end=1)
    rows = {
        s1.id: [DictRow({"x": 1}), DictRow({"x": 2})],
        s2.id: [DictRow({"x": 3})],
    }

    pipeline = (
        RefinerPipeline(source=_FakeReader([s1, s2], rows))
        .map(lambda r: {"x": int(r["x"]) + 1})
        .filter(lambda r: int(r["x"]) % 2 == 0)
    )

    stats = pipeline.launch_local(name="unit-test-local", num_workers=1)

    assert stats.workers == 1
    assert stats.claimed == 2
    assert stats.completed == 2
    assert stats.failed == 0
    assert stats.output_rows == 2


def test_build_cpu_sets_partitions_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "refiner.runtime.cpu.available_cpu_ids",
        lambda: [0, 1, 2, 3, 4, 5],
    )
    sets = build_cpu_sets(num_workers=3, cpus_per_worker=2)
    assert sets == [[0, 1], [2, 3], [4, 5]]


def test_build_cpu_sets_raises_when_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "refiner.runtime.cpu.available_cpu_ids",
        lambda: [0, 1, 2],
    )
    with pytest.raises(ValueError):
        build_cpu_sets(num_workers=2, cpus_per_worker=2)


def test_launch_local_multi_worker_subprocess_with_lambda(tmp_path) -> None:
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text('{"x": 1}\n')
    p2.write_text('{"x": 2}\n')
    pipeline = read_jsonl([str(p1), str(p2)]).map(lambda r: {"x": int(r["x"]) + 10})

    stats = pipeline.launch_local(
        name="unit-test-local-subprocess",
        num_workers=2,
        workdir=str(tmp_path),
    )
    assert stats.workers == 2
    assert stats.claimed == 2
    assert stats.completed == 2
    assert stats.failed == 0
    assert stats.output_rows == 2
