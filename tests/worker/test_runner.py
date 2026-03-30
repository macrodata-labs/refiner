from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from collections import defaultdict
from typing import Any, cast

import pyarrow as pa
import pytest

from refiner.pipeline.data.shard import Shard
from refiner import register_gauge
from refiner.pipeline import RefinerPipeline
from refiner.pipeline.expressions import col
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
from refiner.worker.metrics.api import log_gauge
from refiner.platform.client.models import FinalizedShardWorker
from refiner.worker.config import WorkerConfig


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
        self.worker_id = "local"
        self._remaining = list(shards)
        self.claim_previous: list[Shard | None] = []
        self.completed_ids: list[str] = []
        self.failed_ids: list[str] = []
        self.failed_errors: list[str | None] = []
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
        self.failed_ids.append(shard.id)
        self.failed_errors.append(error)

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

    def register_user_gauge(self, **kwargs) -> None:
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


class _CloseFailingSink(_RecordingSink):
    def close(self) -> None:
        raise RuntimeError("close failed")


class _ShortWriteAndCloseFailingSink(_CloseFailingSink):
    def write_block(self, block) -> dict[str, int]:
        counts = super().write_block(block)
        return {shard_id: 1 for shard_id in counts}


class _FlushFailingTelemetryEmitter(_NoopTelemetryEmitter):
    def force_flush_user_metrics(self) -> None:
        raise RuntimeError("flush failed")


class _RecordingTelemetryEmitter(_NoopTelemetryEmitter):
    def __init__(self) -> None:
        self.log_flushes = 0

    def force_flush_logs(self) -> None:
        self.log_flushes += 1


class _MetricRecordingTelemetryEmitter(_NoopTelemetryEmitter):
    def __init__(self) -> None:
        self.counters: list[dict[str, Any]] = []
        self.histograms: list[dict[str, Any]] = []
        self.gauges: list[dict[str, Any]] = []
        self.registered_gauges: list[dict[str, Any]] = []

    def emit_user_counter(self, **kwargs) -> None:
        self.counters.append(kwargs)

    def emit_user_histogram(self, **kwargs) -> None:
        self.histograms.append(kwargs)

    def emit_user_gauge(self, **kwargs) -> None:
        self.gauges.append(kwargs)

    def register_user_gauge(self, **kwargs) -> None:
        self.registered_gauges.append(kwargs)


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


def _run_local_worker(
    *,
    rows_by_shard: Mapping[str, Sequence[Row]],
    runtime_lifecycle: _FakeRuntimeLifecycle,
    sink: BaseSink | None = None,
    transform=None,
) -> Worker:
    pipeline = RefinerPipeline(source=_FakeReader(rows_by_shard))
    if transform is not None:
        pipeline = pipeline.map(transform)
    if sink is not None:
        pipeline = pipeline.with_sink(sink)

    worker = Worker(
        pipeline=pipeline,
        run_handle=_local_run(),
    )
    cast(Any, worker)._start_local_session = lambda: (
        runtime_lifecycle,
        _local_run().with_worker(worker_id=runtime_lifecycle.worker_id),
    )
    return worker


def _steps_by_label(metrics: list[dict[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for metric in metrics:
        out[cast(str, metric["label"])].add(cast(int, metric["step_index"]))
    return {label: sorted(steps) for label, steps in out.items()}


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


def test_platform_worker_start_reports_worker_config() -> None:
    shard = _shard("input.jsonl", 0, 1)
    seen: dict[str, Any] = {}

    class _RecordingClient:
        base_url = "https://example.com"
        api_key = "md_test"

        def report_worker_started(self, **kwargs) -> WorkerStartedResponse:
            seen.update(kwargs)
            return WorkerStartedResponse(worker_id="worker-0")

    worker = Worker(
        pipeline=RefinerPipeline(source=_FakeReader({shard.id: []})),
        run_handle=RunHandle(
            job_id="job-1",
            stage_index=0,
            worker_name="cloud-rank-0",
            worker_config=WorkerConfig(
                cpu_cores=1,
                memory_mb=2048,
                gpu_count=1,
                gpu_type="h100",
            ),
            client=cast(Any, _RecordingClient()),
        ),
    )

    runtime_lifecycle, run = worker._start_platform_session()

    assert runtime_lifecycle.run.worker_id == "worker-0"
    assert run.worker_id == "worker-0"
    assert seen["config"] == WorkerConfig(
        cpu_cores=1,
        memory_mb=2048,
        gpu_count=1,
        gpu_type="h100",
    )


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
    cast(Any, worker)._start_local_session = lambda: (
        runtime_lifecycle,
        _local_run().with_worker(worker_id=runtime_lifecycle.worker_id),
    )

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
    cast(Any, worker)._start_local_session = lambda: (
        runtime_lifecycle,
        _local_run().with_worker(worker_id=runtime_lifecycle.worker_id),
    )

    stats = worker.run()

    assert stats.claimed == 2
    assert stats.completed == 0
    assert stats.failed == 2
    assert runtime_lifecycle.completed_ids == []
    assert runtime_lifecycle.failed_ids == [shard1.id, shard2.id]
    assert runtime_lifecycle.failed_errors == ["kaboom", "kaboom"]


def test_worker_failure_uses_exception_type_when_message_is_empty() -> None:
    shard = _shard("boom", 0, 1)
    runtime_lifecycle = _FakeRuntimeLifecycle([shard])
    rows_by_shard = {shard.id: [DictRow({"x": 1})]}

    def fail(row: Row) -> Row:
        del row
        raise RuntimeError()

    worker = Worker(
        pipeline=RefinerPipeline(source=_FakeReader(rows_by_shard)).map(fail),
        run_handle=_local_run(),
    )
    cast(Any, worker)._start_local_session = lambda: (
        runtime_lifecycle,
        _local_run().with_worker(worker_id=runtime_lifecycle.worker_id),
    )

    stats = worker.run()

    assert stats.failed == 1
    assert runtime_lifecycle.failed_errors == ["RuntimeError"]


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
    cast(Any, worker)._start_local_session = lambda: (
        runtime_lifecycle,
        _local_run().with_worker(worker_id=runtime_lifecycle.worker_id),
    )
    stats = worker.run()

    assert stats.claimed == 2
    assert stats.completed == 2
    assert emitted == [shard2.id, shard1.id]


def test_worker_runtime_complete_errors_fail_the_shard_without_crashing() -> None:
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
    cast(Any, worker)._start_local_session = lambda: (
        runtime_lifecycle,
        _local_run().with_worker(worker_id=runtime_lifecycle.worker_id),
    )
    stats = worker.run()

    assert stats.claimed == 1
    assert stats.completed == 0
    assert stats.failed == 1
    assert runtime_lifecycle.completed_ids == []
    assert runtime_lifecycle.failed_ids == [shard.id]
    assert runtime_lifecycle.failed_errors == ["complete failed"]


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
    cast(Any, worker)._start_local_session = lambda: (
        runtime_lifecycle,
        _local_run().with_worker(worker_id=runtime_lifecycle.worker_id),
    )

    stats = worker.run()

    assert stats.completed == 1
    assert sum(batch.get(shard.id, 0) for batch in sink.written_counts) == 2
    assert sink.completed_shards == [shard.id]
    assert runtime_lifecycle.completed_ids == [shard.id]


def test_worker_metrics_use_correct_step_indexes_for_all_block_types(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    shard = _shard("p", 0, 3)
    rows_by_shard = {
        shard.id: [
            DictRow({"x": 1, "raw": " a ", "dropme": "x"}),
            DictRow({"x": 2, "raw": " b ", "dropme": "y"}),
            DictRow({"x": 3, "raw": " c ", "dropme": "z"}),
        ],
    }
    runtime_lifecycle = _FakeRuntimeLifecycle([shard])
    emitter = _MetricRecordingTelemetryEmitter()
    monkeypatch.setattr("refiner.worker.runner.NOOP_USER_METRICS_EMITTER", emitter)

    async def add_async(row: Row) -> dict[str, int]:
        log_gauge("async_gauge", int(row["x"]))
        register_gauge("async_registered", lambda: 1)
        return {"score_int": int(row["score_int"]) + 1}

    pipeline = (
        RefinerPipeline(source=_FakeReader(rows_by_shard))
        .with_column("score", col("x") + 10)
        .with_columns(clean=col("raw").str.strip(), keep=col("x") >= 2)
        .filter(col("keep"))
        .map_table(
            lambda table: (
                log_gauge("table_gauge", table.num_rows),
                register_gauge("table_registered", lambda: table.num_rows),
                table.append_column("mapped", pa.array(["m"] * table.num_rows)),
            )[2]
        )
        .select("x", "score", "clean", "dropme", "mapped")
        .drop("dropme")
        .rename(score="score_int")
        .cast(score_int="int64")
        .map(
            lambda row: (
                log_gauge("map_gauge", int(row["x"])),
                register_gauge("map_registered", lambda: 1),
                row.log_histogram("map_hist", int(row["x"]), unit="rows"),
                {"score_int": int(row["score_int"]) * 2},
            )[3]
        )
        .map_async(add_async)
        .filter(lambda row: int(row["x"]) == 2)
        .flat_map(
            lambda row: (
                log_gauge("flat_map_gauge", 2),
                register_gauge("flat_map_registered", lambda: 2),
                row.log_throughput("flat_map_counter", 1, unit="rows"),
                [{"variant": "base"}, {"variant": "extra"}],
            )[3]
        )
        .batch_map(
            lambda rows: (
                log_gauge("batch_gauge", len(rows)),
                register_gauge("batch_registered", lambda: len(rows)),
                rows,
            )[2],
            batch_size=2,
        )
        .write_jsonl(tmp_path / "out")
    )

    worker = Worker(
        pipeline=pipeline,
        run_handle=_local_run(),
    )
    cast(Any, worker)._start_local_session = lambda: (
        runtime_lifecycle,
        _local_run().with_worker(worker_id=runtime_lifecycle.worker_id),
    )

    stats = worker.run()

    assert stats.completed == 1
    assert stats.output_rows == 2

    counter_steps_by_label = _steps_by_label(emitter.counters)
    histogram_steps_by_label = _steps_by_label(emitter.histograms)
    gauge_steps_by_label = _steps_by_label(emitter.gauges)
    registered_gauge_steps_by_label = _steps_by_label(emitter.registered_gauges)

    assert counter_steps_by_label["rows_read"] == [0]
    assert counter_steps_by_label["rows_processed"] == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ]
    assert counter_steps_by_label["rows_kept"] == [3, 11]
    assert counter_steps_by_label["rows_dropped"] == [3, 11]
    assert counter_steps_by_label["flat_map_counter"] == [12]
    assert counter_steps_by_label["rows_out"] == [13]
    assert counter_steps_by_label["rows_written"] == [14]
    assert counter_steps_by_label["files_written"] == [14]
    assert histogram_steps_by_label["map_hist"] == [9]
    assert histogram_steps_by_label["rows_out"] == [12]
    assert gauge_steps_by_label["table_gauge"] == [4]
    assert gauge_steps_by_label["map_gauge"] == [9]
    assert gauge_steps_by_label["async_gauge"] == [10]
    assert gauge_steps_by_label["flat_map_gauge"] == [12]
    assert gauge_steps_by_label["batch_gauge"] == [13]
    assert registered_gauge_steps_by_label["table_registered"] == [4]
    assert registered_gauge_steps_by_label["map_registered"] == [9]
    assert registered_gauge_steps_by_label["async_registered"] == [10]
    assert registered_gauge_steps_by_label["flat_map_registered"] == [12]
    assert registered_gauge_steps_by_label["batch_registered"] == [13]


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


def test_worker_flushes_logs_on_failure(monkeypatch) -> None:
    shard = _shard("boom", 0, 1)
    rows_by_shard = {shard.id: [DictRow({"x": 2})]}
    emitter = _RecordingTelemetryEmitter()

    def maybe_fail(row: Row) -> Row:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(
        "refiner.worker.runner.OtelTelemetryEmitter", lambda **_: emitter
    )
    worker = Worker(
        pipeline=RefinerPipeline(source=_FakeReader(rows_by_shard)).map(maybe_fail),
        run_handle=RunHandle(
            job_id="job",
            stage_index=0,
            client=cast(Any, _LifecycleClientWithFailingTelemetry(shard)),
            worker_name="worker-0",
        ),
    )

    stats = worker.run()

    assert stats.failed == 1
    assert emitter.log_flushes == 2


def test_worker_suppresses_sink_close_errors_after_execution_failure() -> None:
    shard = _shard("boom", 0, 1)
    runtime_lifecycle = _FakeRuntimeLifecycle([shard])
    rows_by_shard = {shard.id: [DictRow({"x": 2})]}

    def maybe_fail(row: Row) -> Row:
        raise RuntimeError("kaboom")

    stats = _run_local_worker(
        rows_by_shard=rows_by_shard,
        runtime_lifecycle=runtime_lifecycle,
        sink=_CloseFailingSink(),
        transform=maybe_fail,
    )
    stats = stats.run()

    assert stats.failed == 1
    assert runtime_lifecycle.failed_ids == [shard.id]
    assert runtime_lifecycle.failed_errors == ["kaboom"]


def test_worker_suppresses_sink_close_errors_after_run_failure() -> None:
    shard = _shard("p", 0, 1)
    runtime_lifecycle = _FakeRuntimeLifecycle([shard])
    rows_by_shard = {shard.id: [DictRow({"x": 1}), DictRow({"x": 2})]}

    worker = _run_local_worker(
        rows_by_shard=rows_by_shard,
        runtime_lifecycle=runtime_lifecycle,
        sink=_ShortWriteAndCloseFailingSink(),
    )

    with pytest.raises(RuntimeError, match="worker finished with unflushed shards"):
        worker.run()


def test_worker_raises_sink_close_errors_after_success() -> None:
    shard = _shard("p", 0, 1)
    runtime_lifecycle = _FakeRuntimeLifecycle([shard])
    rows_by_shard = {shard.id: [DictRow({"x": 1})]}

    worker = _run_local_worker(
        rows_by_shard=rows_by_shard,
        runtime_lifecycle=runtime_lifecycle,
        sink=_CloseFailingSink(),
    )

    with pytest.raises(RuntimeError, match="close failed"):
        worker.run()


__all__: list[str] = []
