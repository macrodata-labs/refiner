from __future__ import annotations

from refiner.pipeline import task
from refiner.pipeline.planning import compile_pipeline_plan


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


def test_task_wraps_scalar_return_as_result() -> None:
    pipeline = task(lambda rank, _world_size: rank, num_tasks=3)

    out = list(pipeline.iter_rows())

    assert [int(row["task_rank"]) for row in out] == [0, 1, 2]
    assert [int(row["result"]) for row in out] == [0, 1, 2]


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
