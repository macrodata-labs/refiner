from __future__ import annotations

from collections.abc import Iterable, Iterator

from refiner.ledger.backend.base import BaseLedger, LedgerConfig
from refiner.ledger.shard import Shard
from refiner.pipeline import RefinerPipeline
from refiner.readers.base import BaseReader
from refiner.readers.row import DictRow, Row
from refiner.worker import Worker


class _FakeReader(BaseReader):
    def __init__(self, rows_by_shard_id: dict[str, list[Row]]):
        self.rows_by_shard_id = rows_by_shard_id

    def list_shards(self) -> list[Shard]:
        # Not needed by worker tests, but required by BaseReader contract.
        return []

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        for row in self.rows_by_shard_id.get(shard.id, []):
            yield row


class _FakeLedger(BaseLedger):
    def __init__(self, shards: list[Shard]):
        super().__init__(run_id="run", worker_id=1, config=LedgerConfig())
        self._remaining = list(shards)
        self.claim_previous: list[Shard | None] = []
        self.completed_ids: list[str] = []
        self.failed_ids: list[str] = []
        self.heartbeat_ids: list[str] = []

    def seed_shards(self, shards: Iterable[Shard]) -> None:
        self._remaining = list(shards)

    def claim(self, previous: Shard | None = None) -> Shard | None:
        self.claim_previous.append(previous)
        if not self._remaining:
            return None
        return self._remaining.pop(0)

    def heartbeat(self, shard: Shard) -> None:
        self.heartbeat_ids.append(shard.id)

    def complete(self, shard: Shard) -> None:
        self.completed_ids.append(shard.id)

    def fail(self, shard: Shard, error: str | None = None) -> None:
        self.failed_ids.append(shard.id)


def test_pipeline_executes_row_and_batch_steps() -> None:
    source_rows = [
        DictRow({"x": 1}),
        DictRow({"x": 2}),
        DictRow({"x": 3}),
        DictRow({"x": 4}),
    ]
    pipeline = (
        RefinerPipeline(source=_FakeReader({}))
        .map(lambda r: {"x": r["x"] + 1})
        .map(lambda r: None if r["x"] % 2 == 0 else r)
        .map(lambda rows: [row for row in rows if row["x"] >= 3], batch_size=2)
        .map(lambda r: {"y": r["x"] * 10})
    )

    out = list(pipeline.execute_rows(source_rows))

    assert [r["x"] for r in out] == [3, 5]
    assert [r["y"] for r in out] == [30, 50]


def test_worker_runs_fused_pipeline_and_updates_ledger() -> None:
    shard1 = Shard(path="p1", start=0, end=10)
    shard2 = Shard(path="p2", start=0, end=10)
    ledger = _FakeLedger([shard1, shard2])

    rows_by_shard = {
        shard1.id: [
            DictRow({"sid": shard1.id, "x": 1}),
            DictRow({"sid": shard1.id, "x": 2}),
        ],
        shard2.id: [DictRow({"sid": shard2.id, "x": 10})],
    }
    emitted: list[tuple[str, int]] = []

    def tap(row: Row) -> Row:
        emitted.append((row["sid"], row["x"]))
        return row

    pipeline = (
        RefinerPipeline(source=_FakeReader(rows_by_shard))
        .map(lambda r: {"x": r["x"] + 1})
        .map(lambda rows: list(reversed(rows)), batch_size=2)
        .map(tap)
    )

    worker = Worker(
        rank=0,
        ledger=ledger,
        pipeline=pipeline,
        heartbeat_every_rows=1,
    )

    stats = worker.run()

    assert stats.claimed == 2
    assert stats.completed == 2
    assert stats.failed == 0
    assert stats.output_rows == 3
    assert ledger.completed_ids == [shard1.id, shard2.id]
    assert ledger.failed_ids == []
    # Previous hint should be used in claims after first completion.
    assert ledger.claim_previous[0] is None
    assert ledger.claim_previous[1] == shard1
    assert emitted == [(shard1.id, 3), (shard1.id, 2), (shard2.id, 11)]


def test_worker_fails_entire_claimed_group_on_exception() -> None:
    shard1 = Shard(path="ok", start=0, end=1)
    shard2 = Shard(path="boom", start=0, end=1)
    ledger = _FakeLedger([shard1, shard2])

    rows_by_shard = {
        shard1.id: [DictRow({"x": 1})],
        shard2.id: [DictRow({"x": 2})],
    }

    def maybe_fail(row: Row) -> Row | None:
        if row["x"] == 2:
            raise RuntimeError("kaboom")
        return row

    pipeline = RefinerPipeline(source=_FakeReader(rows_by_shard)).map(maybe_fail)
    worker = Worker(rank=0, ledger=ledger, pipeline=pipeline)

    stats = worker.run()

    assert stats.claimed == 2
    assert stats.completed == 0
    assert stats.failed == 2
    assert ledger.completed_ids == []
    assert ledger.failed_ids == [shard1.id, shard2.id]


def test_worker_can_batch_across_shards() -> None:
    shard1 = Shard(path="s1", start=0, end=1)
    shard2 = Shard(path="s2", start=0, end=1)
    ledger = _FakeLedger([shard1, shard2])

    rows_by_shard = {
        shard1.id: [DictRow({"sid": shard1.id, "x": 1})],
        shard2.id: [DictRow({"sid": shard2.id, "x": 2})],
    }
    emitted: list[str] = []

    def tap(row: Row) -> Row:
        emitted.append(row["sid"])
        return row

    pipeline = (
        RefinerPipeline(source=_FakeReader(rows_by_shard))
        .map(lambda rows: list(reversed(rows)), batch_size=2)
        .map(tap)
    )

    worker = Worker(
        rank=0,
        ledger=ledger,
        pipeline=pipeline,
    )
    stats = worker.run()

    assert stats.claimed == 2
    assert stats.completed == 2
    # batch reverse should cross shard boundary and invert shard emission order.
    assert emitted == [shard2.id, shard1.id]


# Keep pytest from treating imported typing names as tests on some plugins.
__all__: list[str] = []
