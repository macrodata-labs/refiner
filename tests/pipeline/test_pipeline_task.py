from __future__ import annotations

import pytest

from refiner.pipeline import RefinerPipeline, task
from refiner.pipeline.planning import compile_pipeline_plan
from refiner.pipeline.sources.items import ItemsSource


def test_task_invokes_fn_with_rank_and_world_size() -> None:
    seen: list[tuple[int, int]] = []

    def worker_fn(rank: int, world_size: int):
        seen.append((rank, world_size))
        return {"rank": rank, "world_size": world_size}

    pipeline = task(worker_fn, num_tasks=4)
    out = list(pipeline.iter_rows())

    assert seen == [(0, 4), (1, 4), (2, 4), (3, 4)]
    assert [int(row["rank"]) for row in out] == [0, 1, 2, 3]
    assert all(int(row["world_size"]) == 4 for row in out)


def test_task_defaults_to_one_in_flight_shard() -> None:
    pipeline = task(lambda rank, _world_size: rank, num_tasks=2)

    assert pipeline.max_in_flight_shards == 1
    assert RefinerPipeline(ItemsSource(items=[1])).max_in_flight_shards is None


def test_task_allows_explicit_in_flight_shard_limit() -> None:
    pipeline = task(lambda rank, _world_size: rank, num_tasks=2)

    configured = pipeline.with_max_in_flight_shards(2)

    assert configured.max_in_flight_shards == 2
    assert pipeline.max_in_flight_shards == 1


def test_in_flight_shard_limit_must_be_positive() -> None:
    pipeline = task(lambda rank, _world_size: rank, num_tasks=1)

    with pytest.raises(ValueError, match="max_in_flight_shards must be > 0"):
        pipeline.with_max_in_flight_shards(0)


def test_task_wraps_scalar_return_as_result() -> None:
    pipeline = task(lambda rank, _world_size: rank, num_tasks=3)

    out = list(pipeline.iter_rows())

    assert [int(row["task_rank"]) for row in out] == [0, 1, 2]
    assert [int(row["result"]) for row in out] == [0, 1, 2]


def test_task_treats_binary_buffers_as_scalar_results() -> None:
    for value in (bytearray(b"ok"), memoryview(b"ok")):
        pipeline = task(lambda _rank, _world_size, value=value: value, num_tasks=1)

        out = list(pipeline.iter_rows())

        assert len(out) == 1
        assert out[0]["result"] == value


def test_task_allows_no_return_for_side_effect_only_work() -> None:
    seen: list[int] = []

    def worker_fn(rank: int, _world_size: int) -> None:
        seen.append(rank)

    pipeline = task(worker_fn, num_tasks=3)

    assert list(pipeline.iter_rows()) == []
    assert seen == [0, 1, 2]


def test_task_allows_yielding_multiple_rows() -> None:
    def worker_fn(rank: int, world_size: int):
        yield {"rank": rank, "phase": "start"}
        yield {"rank": rank, "phase": "done", "world_size": world_size}

    pipeline = task(worker_fn, num_tasks=2)

    out = [row.to_dict() for row in pipeline.iter_rows()]

    assert out == [
        {"task_rank": 0, "rank": 0, "phase": "start"},
        {"task_rank": 0, "rank": 0, "phase": "done", "world_size": 2},
        {"task_rank": 1, "rank": 1, "phase": "start"},
        {"task_rank": 1, "rank": 1, "phase": "done", "world_size": 2},
    ]


def test_task_compiles_source_and_task_step_plan() -> None:
    pipeline = task(lambda rank, world_size: {"ok": rank < world_size}, num_tasks=3)
    payload = compile_pipeline_plan(pipeline)
    steps = payload["stages"][0]["steps"]
    assert steps[0]["name"] == "task"
    assert steps[0]["args"]["num_tasks"] == 3
    assert steps[1]["name"] == "task_2"
    assert steps[1]["type"] == "flat_map"
    assert "fn" in steps[1]["args"]
    assert steps[1]["args"]["__meta"]["fn"] == "code"
