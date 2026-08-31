from __future__ import annotations

from collections.abc import Iterable

import cloudpickle
import pyarrow as pa
import pytest

from refiner.pipeline import from_items
from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import compile_pipeline_plan


def test_map_table_factory_is_deferred_and_reused_per_execution() -> None:
    factory_calls = 0
    mapper_calls: list[int] = []

    class Mapper:
        def __call__(self, table: pa.Table) -> pa.Table:
            mapper_calls.append(table.num_rows)
            return table.append_column(
                "call_rows",
                pa.array([table.num_rows] * table.num_rows),
            )

    def create_mapper() -> Mapper:
        nonlocal factory_calls
        factory_calls += 1
        return Mapper()

    pipeline = (
        from_items([{"x": value} for value in range(5)], items_per_shard=2)
        .with_max_block_rows(2)
        .map_table(factory=create_mapper)
    )

    assert pipeline.output_schema() is None
    plan = compile_pipeline_plan(pipeline)
    step_args = plan["stages"][0]["steps"][1]["args"]
    assert "factory" in step_args
    assert "fn" not in step_args
    cloudpickle.dumps(pipeline)
    assert factory_calls == 0

    rows = list(pipeline.iter_rows())

    assert factory_calls == 1
    assert mapper_calls == [2, 2, 1]
    assert [int(row["call_rows"]) for row in rows] == [2, 2, 2, 2, 1]

    list(pipeline.iter_rows())
    assert factory_calls == 2
    assert mapper_calls == [2, 2, 1, 2, 2, 1]


def test_batch_map_factory_is_reused_across_batches_and_shards() -> None:
    instances: list[BatchMapper] = []

    class BatchMapper:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def __call__(self, rows: list[Row]) -> Iterable[Row]:
            self.batch_sizes.append(len(rows))
            return [row.update({"batch_size": len(rows)}) for row in rows]

    def create_mapper() -> BatchMapper:
        mapper = BatchMapper()
        instances.append(mapper)
        return mapper

    pipeline = from_items(
        [{"x": value} for value in range(5)], items_per_shard=1
    ).batch_map(factory=create_mapper, batch_size=2)

    rows = list(pipeline.iter_rows())

    assert len(instances) == 1
    assert instances[0].batch_sizes == [2, 2, 1]
    assert [int(row["batch_size"]) for row in rows] == [2, 2, 2, 2, 1]


def test_map_async_factory_is_reused_and_closed() -> None:
    instances: list[AsyncMapper] = []

    class AsyncMapper:
        def __init__(self) -> None:
            self.values: list[int] = []
            self.close_calls = 0

        async def __call__(self, row: Row) -> dict[str, int]:
            value = int(row["x"])
            self.values.append(value)
            return {"mapped": value + 1}

        async def aclose(self) -> None:
            self.close_calls += 1

    def create_mapper() -> AsyncMapper:
        mapper = AsyncMapper()
        instances.append(mapper)
        return mapper

    pipeline = from_items(
        [{"x": value} for value in range(4)], items_per_shard=1
    ).map_async(
        factory=create_mapper,
        max_in_flight=2,
    )

    rows = list(pipeline.iter_rows())

    assert [int(row["mapped"]) for row in rows] == [1, 2, 3, 4]
    assert len(instances) == 1
    assert instances[0].values == [0, 1, 2, 3]
    assert instances[0].close_calls == 1


def test_factory_initialization_failure_closes_earlier_async_callable() -> None:
    instances: list[AsyncMapper] = []

    class AsyncMapper:
        def __init__(self) -> None:
            self.close_calls = 0

        async def __call__(self, row: Row) -> Row:
            return row

        async def aclose(self) -> None:
            self.close_calls += 1

    def create_mapper() -> AsyncMapper:
        mapper = AsyncMapper()
        instances.append(mapper)
        return mapper

    pipeline = (
        from_items([{"x": 1}])
        .map_async(factory=create_mapper)
        .map_async(factory=lambda: object())
    )

    with pytest.raises(TypeError, match="factory must return a callable"):
        list(pipeline.iter_rows())

    assert len(instances) == 1
    assert instances[0].close_calls == 1


@pytest.mark.parametrize(
    "operation", ["map_table", "batch_map", "batch_map_async", "map_async"]
)
def test_worker_factory_must_return_callable(operation: str) -> None:
    pipeline = from_items([{"x": 1}])
    if operation == "map_table":
        pipeline = pipeline.map_table(factory=lambda: object())
    elif operation == "batch_map":
        pipeline = pipeline.batch_map(factory=lambda: object(), batch_size=2)
    elif operation == "batch_map_async":
        pipeline = pipeline.batch_map_async(factory=lambda: object(), batch_size=2)
    else:
        pipeline = pipeline.map_async(factory=lambda: object())

    with pytest.raises(TypeError, match="factory must return a callable"):
        list(pipeline.iter_rows())


def test_transforms_require_exactly_one_callback_or_factory() -> None:
    pipeline = from_items([{"x": 1}])

    with pytest.raises(ValueError, match="exactly one of fn or factory"):
        pipeline.map_table()
    with pytest.raises(ValueError, match="exactly one of fn or factory"):
        pipeline.map_table(lambda table: table, factory=lambda: lambda table: table)

    with pytest.raises(ValueError, match="exactly one of fn or factory"):
        pipeline.batch_map(batch_size=2)
    with pytest.raises(ValueError, match="exactly one of fn or factory"):
        pipeline.batch_map(
            lambda rows: rows,
            factory=lambda: lambda rows: rows,
            batch_size=2,
        )

    with pytest.raises(ValueError, match="exactly one of fn or factory"):
        pipeline.batch_map_async(batch_size=2)
    with pytest.raises(ValueError, match="exactly one of fn or factory"):
        pipeline.batch_map_async(
            lambda rows: rows,
            factory=lambda: lambda rows: rows,
            batch_size=2,
        )

    with pytest.raises(ValueError, match="exactly one of fn or factory"):
        pipeline.map_async()
    with pytest.raises(ValueError, match="exactly one of fn or factory"):
        pipeline.map_async(lambda row: row, factory=lambda: lambda row: row)
