from __future__ import annotations

import asyncio
from collections.abc import Iterable

import pytest

from refiner.execution.operators.row import execute_row_steps
from refiner.pipeline import from_items
from refiner.pipeline.data.row import DictRow, Row
from refiner.pipeline.planning import compile_pipeline_plan
from refiner.pipeline.steps import FnAsyncBatchStep


def test_async_batch_map_is_bounded_and_preserves_order() -> None:
    active = 0
    max_active = 0
    batch_sizes: list[int] = []

    async def mapper(rows: list[Row]) -> Iterable[Row]:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        values = [int(row["x"]) for row in rows]
        batch_sizes.append(len(rows))
        await asyncio.sleep(0.03 if values[0] == 0 else 0.005)
        active -= 1
        return [row.update({"mapped": int(row["x"]) + 10}) for row in rows]

    pipeline = from_items([{"x": value} for value in range(5)]).batch_map_async(
        mapper,
        batch_size=2,
        max_in_flight=2,
    )

    rows = list(pipeline.iter_rows())

    assert max_active == 2
    assert sorted(batch_sizes) == [1, 2, 2]
    assert [int(row["mapped"]) for row in rows] == [10, 11, 12, 13, 14]


def test_async_batch_map_can_emit_completion_order() -> None:
    async def mapper(rows: list[Row]) -> Iterable[Row]:
        first = int(rows[0]["x"])
        await asyncio.sleep(0.03 if first == 0 else 0.001)
        return rows

    rows = list(
        from_items([{"x": value} for value in range(6)])
        .batch_map_async(
            mapper,
            batch_size=2,
            max_in_flight=2,
            preserve_order=False,
        )
        .iter_rows()
    )

    assert [int(row["x"]) for row in rows[:2]] == [2, 3]
    assert sorted(int(row["x"]) for row in rows) == list(range(6))


def test_async_batch_factory_is_deferred_reused_and_closed() -> None:
    instances: list[Mapper] = []

    class Mapper:
        def __init__(self) -> None:
            self.calls = 0
            self.close_calls = 0

        async def __call__(self, rows: list[Row]) -> Iterable[Row]:
            self.calls += 1
            return rows

        async def aclose(self) -> None:
            self.close_calls += 1

    def create_mapper() -> Mapper:
        mapper = Mapper()
        instances.append(mapper)
        return mapper

    pipeline = from_items([{"x": value} for value in range(5)]).batch_map_async(
        factory=create_mapper,
        batch_size=2,
        max_in_flight=2,
    )
    plan = compile_pipeline_plan(pipeline)
    args = plan["stages"][0]["steps"][1]["args"]

    assert args["batch_size"] == 2
    assert args["max_in_flight"] == 2
    assert "factory" in args
    assert "fn" not in args
    assert instances == []

    assert len(list(pipeline.iter_rows())) == 5
    assert len(instances) == 1
    assert instances[0].calls == 3
    assert instances[0].close_calls == 1


def test_async_batch_map_updates_shard_cardinality_on_completion() -> None:
    deltas: list[dict[str, int]] = []

    async def keep_first(rows: list[Row]) -> Iterable[Row]:
        return rows[:1]

    output = list(
        execute_row_steps(
            [
                DictRow({"x": 1}, shard_id="s1"),
                DictRow({"x": 2}, shard_id="s1"),
            ],
            [
                FnAsyncBatchStep(
                    fn=keep_first,
                    index=1,
                    batch_size=2,
                    max_in_flight=2,
                )
            ],
            on_shard_delta=deltas.append,
        )
    )

    assert [int(row["x"]) for row in output] == [1]
    assert deltas == [{"s1": -1}]


def test_async_batch_map_rejects_non_row_output() -> None:
    async def invalid(_rows: list[Row]) -> Iterable[Row]:
        return [{"x": 1}]  # type: ignore[list-item]

    pipeline = from_items([{"x": 1}, {"x": 2}]).batch_map_async(
        invalid,
        batch_size=2,
    )

    with pytest.raises(TypeError, match="batch_map_async.*dict"):
        list(pipeline.iter_rows())


@pytest.mark.parametrize("max_in_flight", [0, -1])
def test_async_batch_map_requires_positive_window(max_in_flight: int) -> None:
    async def mapper(rows: list[Row]) -> Iterable[Row]:
        return rows

    with pytest.raises(ValueError, match="max_in_flight must be > 0"):
        from_items([{"x": 1}]).batch_map_async(
            mapper,
            batch_size=2,
            max_in_flight=max_in_flight,
        )
