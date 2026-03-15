from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

import pytest

from refiner.pipeline.data.shard import Shard
from refiner.pipeline import RefinerPipeline
from refiner.execution.engine import iter_rows
from refiner.pipeline.sinks import BaseSink
from refiner.platform.client import (
    OkResponse,
    RunHandle,
    ShardClaimResponse,
    SerializedShard,
    WorkerStartedResponse,
)
from refiner.pipeline.data.shard import FilePart
from refiner.worker.runner import Worker
from refiner.pipeline.sources.readers.base import BaseReader
from refiner.pipeline.data.row import DictRow, Row
from refiner.platform.client.models import FinalizedShardWorker


class _FakeReader(BaseReader):
    def __init__(self, rows_by_shard_id: Mapping[str, Sequence[Row]]):
        self.rows_by_shard_id = rows_by_shard_id

    def list_shards(self) -> list[Shard]:
        return []

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        for row in self.rows_by_shard_id.get(shard.id, []):
            yield row


class _FakeRuntimeLifecycle:
    def __init__(self, shards: list[Shard]):
        self._remaining = list(shards)
        self.claim_previous: list[Shard | None] = []
        self.completed_ids: list[str] = []
        self.failed_ids: list[str] = []
        self.heartbeat_batches: list[list[str]] = []

    def claim(self, previous: Shard | None = None) -> Shard | None:
        self.claim_previous.append(previous)
        if not self._remaining:
            return None
        return self._remaining.pop(0)

    def heartbeat(self, shards: list[Shard]) -> None:
        self.heartbeat_batches.append([shard.id for shard in shards])

    def complete(self, shard: Shard) -> None:
        self.completed_ids.append(shard.id)

    def fail(self, shard: Shard, error: str | None = None) -> None:
        del error
        self.failed_ids.append(shard.id)

    def finalized_workers(
        self, *, stage_index: int | None = None
    ) -> list[FinalizedShardWorker]:
        del stage_index
        return []


def _shard(path: str, start: int, end: int) -> Shard:
    return Shard.from_file_parts([FilePart(path=path, start=start, end=end)])


def _local_run() -> RunHandle:
    return RunHandle(job_id="job", stage_index=0)


class _NoopTelemetryEmitter:
    def emit_user_counter(self, **kwargs) -> None:
        del kwargs

    def emit_user_gauge(self, **kwargs) -> None:
        del kwargs

    def emit_user_histogram(self, **kwargs) -> None:
        del kwargs

    def force_flush_user_metrics(self) -> None:
        return None

    def force_flush_resource_metrics(self) -> None:
        return None

    def force_flush_logs(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class _RecordingSink(BaseSink):
    def __init__(self) -> None:
        self.written_counts: list[dict[str, int]] = []
        self.completed_shards: list[str] = []

    def write_block(self, block) -> dict[str, int]:
        if isinstance(block, list):
            counts: dict[str, int] = {}
            for row in block:
                shard_id = row.require_shard_id()
                counts[shard_id] = counts.get(shard_id, 0) + 1
        else:
            counts = {}
            for shard_id in block.column("__shard_id").to_pylist():
                counts[shard_id] = counts.get(shard_id, 0) + 1
        self.written_counts.append(counts)
        return counts

    def on_shard_complete(self, shard_id: str) -> None:
        self.completed_shards.append(shard_id)


class _FlushFailingTelemetryEmitter(_NoopTelemetryEmitter):
    def force_flush_user_metrics(self) -> None:
        raise RuntimeError("flush failed")


class _LifecycleClientWithFailingTelemetry:
    def __init__(self, shard: Shard):
        self._next_shard = shard
        self.base_url = "https://example.com"
        self.api_key = "md_test"

    def report_worker_started(self, **kwargs) -> WorkerStartedResponse:
        del kwargs
        return WorkerStartedResponse(worker_id="worker-0")

    def report_worker_finished(self, **kwargs) -> None:
        del kwargs

    def shard_claim(self, **kwargs):
        del kwargs
        if self._next_shard is None:
            return ShardClaimResponse(shard=None)
        shard = self._next_shard
        self._next_shard = None
        return ShardClaimResponse(
            shard=SerializedShard(
                shard_id=shard.id,
                descriptor=shard.descriptor.to_dict(),
            )
        )

    def shard_heartbeat(self, **kwargs):
        del kwargs
        return OkResponse()

    def shard_finish(self, **kwargs):
        del kwargs
        return OkResponse()


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
        .filter(lambda r: int(r["x"]) % 2 != 0)
        .batch_map(lambda rows: [row for row in rows if row["x"] >= 3], batch_size=2)
        .map(lambda r: {"y": r["x"] * 10})
    )

    out = list(iter_rows(pipeline.execute(source_rows)))

    assert [r["x"] for r in out] == [3, 5]
    assert [r["y"] for r in out] == [30, 50]


def test_worker_runs_fused_pipeline_and_updates_runtime_lifecycle() -> None:
    shard1 = _shard("p1", 0, 10)
    shard2 = _shard("p2", 0, 10)
    runtime_lifecycle = _FakeRuntimeLifecycle([shard1, shard2])

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
        .batch_map(lambda rows: list(reversed(rows)), batch_size=2)
        .map(tap)
    )

    worker = Worker(
        pipeline=pipeline,
        heartbeat_interval_seconds=1,
        run_handle=_local_run(),
    )
    worker._start_local_session = lambda: runtime_lifecycle  # type: ignore[method-assign]

    stats = worker.run()

    assert stats.claimed == 2
    assert stats.completed == 2
    assert stats.failed == 0
    assert stats.output_rows == 3
    assert runtime_lifecycle.completed_ids == [shard1.id, shard2.id]
    assert runtime_lifecycle.failed_ids == []
    assert runtime_lifecycle.claim_previous[0] is None
    assert runtime_lifecycle.claim_previous[1] == shard1
    assert emitted == [(shard1.id, 3), (shard1.id, 2), (shard2.id, 11)]


def test_worker_fails_entire_claimed_group_on_exception() -> None:
    shard1 = _shard("ok", 0, 1)
    shard2 = _shard("boom", 0, 1)
    runtime_lifecycle = _FakeRuntimeLifecycle([shard1, shard2])

    rows_by_shard = {
        shard1.id: [DictRow({"x": 1})],
        shard2.id: [DictRow({"x": 2})],
    }

    def maybe_fail(row: Row) -> Row:
        if row["x"] == 2:
            raise RuntimeError("kaboom")
        return row

    pipeline = RefinerPipeline(source=_FakeReader(rows_by_shard)).map(maybe_fail)
    worker = Worker(
        pipeline=pipeline,
        run_handle=_local_run(),
    )
    worker._start_local_session = lambda: runtime_lifecycle  # type: ignore[method-assign]

    stats = worker.run()

    assert stats.claimed == 2
    assert stats.completed == 0
    assert stats.failed == 2
    assert runtime_lifecycle.completed_ids == []
    assert runtime_lifecycle.failed_ids == [shard1.id, shard2.id]


def test_worker_can_batch_across_shards() -> None:
    shard1 = _shard("s1", 0, 1)
    shard2 = _shard("s2", 0, 1)
    runtime_lifecycle = _FakeRuntimeLifecycle([shard1, shard2])

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
        .batch_map(lambda rows: list(reversed(rows)), batch_size=2)
        .map(tap)
    )

    worker = Worker(
        pipeline=pipeline,
        run_handle=_local_run(),
    )
    worker._start_local_session = lambda: runtime_lifecycle  # type: ignore[method-assign]
    stats = worker.run()

    assert stats.claimed == 2
    assert stats.completed == 2
    assert emitted == [shard2.id, shard1.id]


def test_worker_runtime_complete_errors_are_not_swallowed() -> None:
    shard = _shard("p", 0, 1)
    rows_by_shard = {shard.id: [DictRow({"x": 1})]}
    pipeline = RefinerPipeline(source=_FakeReader(rows_by_shard))

    class _FailingCompleteRuntimeLifecycle(_FakeRuntimeLifecycle):
        def complete(self, shard: Shard) -> None:
            del shard
            raise RuntimeError("complete failed")

    runtime_lifecycle = _FailingCompleteRuntimeLifecycle([shard])
    worker = Worker(
        pipeline=pipeline,
        run_handle=_local_run(),
    )
    worker._start_local_session = lambda: runtime_lifecycle  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="complete failed"):
        worker.run()


def test_worker_completes_shards_only_after_sink_drain() -> None:
    shard = _shard("p", 0, 2)
    rows_by_shard = {
        shard.id: [DictRow({"x": 1}), DictRow({"x": 2})],
    }
    runtime_lifecycle = _FakeRuntimeLifecycle([shard])
    sink = _RecordingSink()

    worker = Worker(
        pipeline=RefinerPipeline(source=_FakeReader(rows_by_shard)).with_sink(sink),
        run_handle=_local_run(),
    )
    worker._start_local_session = lambda: runtime_lifecycle  # type: ignore[method-assign]

    stats = worker.run()

    assert stats.completed == 1
    assert sum(batch.get(shard.id, 0) for batch in sink.written_counts) == 2
    assert sink.completed_shards == [shard.id]
    assert runtime_lifecycle.completed_ids == [shard.id]


def test_worker_shard_flush_errors_are_not_swallowed(monkeypatch) -> None:
    shard = _shard("p", 0, 1)
    rows_by_shard = {shard.id: [DictRow({"x": 1})]}
    pipeline = RefinerPipeline(source=_FakeReader(rows_by_shard))
    monkeypatch.setattr(
        "refiner.worker.runner.OtelTelemetryEmitter",
        lambda **_: _FlushFailingTelemetryEmitter(),
    )

    worker = Worker(
        pipeline=pipeline,
        run_handle=RunHandle(
            job_id="job",
            stage_index=0,
            client=cast(Any, _LifecycleClientWithFailingTelemetry(shard)),
            worker_name="worker-0",
        ),
    )
    with pytest.raises(RuntimeError, match="flush failed"):
        worker.run()


__all__: list[str] = []
