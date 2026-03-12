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


def test_task_compiles_source_and_task_step_plan() -> None:
    pipeline = task(lambda rank, world_size: {"ok": rank < world_size}, num_tasks=3)
    payload = compile_pipeline_plan(pipeline)
    steps = payload["stages"][0]["steps"]
    assert steps[0]["name"] == "task"
    assert steps[0]["args"]["num_tasks"] == 3
    assert steps[1]["name"] == "task_2"
    assert steps[1]["type"] == "row_map"
    assert "fn" in steps[1]["args"]
    assert steps[1]["args"]["__meta"]["fn"] == "code"
