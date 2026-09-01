from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import cloudpickle
import pyarrow as pa
import pytest

import refiner as mdr
from refiner.pipeline import RefinerPipeline, ValidationContract, ValidationError
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.shard import RowRangeDescriptor, Shard, ShardGroupDescriptor
from refiner.pipeline.data.tabular import Tabular
from refiner.execution.engine import iter_rows
from refiner.pipeline.planning import compile_pipeline_plan, plan_pipeline_stages
from refiner.pipeline.sources.base import BaseSource, SourceUnit
from refiner.pipeline.sources.validation import GlobalValidationSource


class _TableSource(BaseSource):
    name = "table_source"

    def __init__(self, table: pa.Table) -> None:
        self.table = table
        self.list_calls = 0

    @property
    def schema(self) -> pa.Schema:
        return self.table.schema

    def list_shards(self) -> list[Shard]:
        self.list_calls += 1
        return [
            Shard.from_row_range(start=0, end=1, global_ordinal=0),
            Shard.from_row_range(start=1, end=2, global_ordinal=1),
        ]

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        assert isinstance(descriptor, RowRangeDescriptor)
        yield Tabular(self.table.slice(descriptor.start, 1))


def _values(pipeline: RefinerPipeline, column: str = "value") -> list[object]:
    return [row[column] for row in pipeline.materialize()]


def test_validate_passes_rows_through_without_changing_arrow_blocks() -> None:
    source = _TableSource(pa.table({"value": [1, 2]}))
    pipeline = RefinerPipeline(source).validate(
        name="values",
        not_null=["value"],
        ranges={"value": (1, 2)},
        predicates={"integer": lambda row: isinstance(row["value"], int)},
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert all(isinstance(block, Tabular) for block in blocks)
    table_blocks = [block for block in blocks if isinstance(block, Tabular)]
    assert [block.table.column("value").to_pylist() for block in table_blocks] == [
        [1],
        [2],
    ]
    assert not isinstance(pipeline.source, GlobalValidationSource)


def test_not_null_reports_contract_rule_and_source_location() -> None:
    pipeline = mdr.from_items([{"value": 1}, {"value": None}]).validate(
        name="training_rows",
        not_null=["value"],
    )

    with pytest.raises(ValidationError) as caught:
        pipeline.materialize()

    error = caught.value
    assert error.contract_name == "training_rows"
    assert error.rule == "not_null:value"
    assert error.location is not None
    assert error.location.source_row_id == 1
    assert "column 'value' contains null" in str(error)


def test_ranges_are_inclusive_and_allow_nulls_unless_required() -> None:
    passing = mdr.from_items(
        [{"score": None}, {"score": 0.0}, {"score": 1.0}]
    ).validate(ranges={"score": (0.0, 1.0)})
    assert _values(passing, "score") == [None, 0.0, 1.0]

    failing = mdr.from_items([{"score": float("nan")}]).validate(
        ranges={"score": (0.0, 1.0)}
    )
    with pytest.raises(ValidationError, match=r"\[range:score\]"):
        failing.materialize()

    incompatible = mdr.from_items([{"score": "high"}]).validate(
        ranges={"score": (0.0, 1.0)}
    )
    with pytest.raises(ValidationError, match="cannot be compared with range"):
        incompatible.materialize()


def test_custom_predicates_are_named_and_wrap_predicate_errors() -> None:
    failing = mdr.from_items([{"value": 3}]).validate(
        predicates={"even": lambda row: int(row["value"]) % 2 == 0}
    )
    with pytest.raises(ValidationError, match=r"\[predicate:even\]"):
        failing.materialize()

    def broken(row: Row) -> bool:
        raise RuntimeError(f"bad value {row['value']}")

    broken_pipeline = mdr.from_items([{"value": 3}]).validate(
        predicates={"safe": broken}
    )
    with pytest.raises(
        ValidationError, match="predicate raised RuntimeError"
    ) as caught:
        broken_pipeline.materialize()
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_unique_is_global_across_source_shards_and_supports_nested_values() -> None:
    pipeline = mdr.from_items(
        [{"key": [1, 2]}, {"key": [3]}, {"key": [1, 2]}],
        items_per_shard=1,
    ).validate(name="keys", unique=["key"])

    assert isinstance(pipeline.source, GlobalValidationSource)
    assert len(pipeline.list_shards()) == 1
    with pytest.raises(ValidationError) as caught:
        pipeline.materialize()

    assert caught.value.rule == "unique:key"
    assert "duplicate value [1, 2]" in str(caught.value)
    assert "first observed" in str(caught.value)


def test_single_column_rules_accept_a_column_name_string() -> None:
    pipeline = mdr.from_items([{"key": 1}, {"key": 1}]).validate(unique="key")

    with pytest.raises(ValidationError, match=r"\[unique:key\]"):
        pipeline.materialize()


def test_unique_together_checks_composite_keys() -> None:
    pipeline = mdr.from_items(
        [
            {"tenant": "a", "external_id": 1},
            {"tenant": "b", "external_id": 1},
            {"tenant": "a", "external_id": 1},
        ]
    ).validate(unique_together=[("tenant", "external_id")])

    with pytest.raises(
        ValidationError,
        match=r"\[unique_together:tenant,external_id\]",
    ):
        pipeline.materialize()


@pytest.mark.parametrize(
    ("rows", "rules", "rule"),
    [
        ([], {"min_rows": 1}, "min_rows"),
        ([{"value": 1}], {"exact_rows": 2}, "exact_rows"),
        ([{"value": 1}, {"value": 2}], {"max_rows": 1}, "max_rows"),
    ],
)
def test_row_count_contracts_cover_empty_and_non_empty_inputs(
    rows: list[dict[str, int]],
    rules: dict[str, int],
    rule: str,
) -> None:
    pipeline = mdr.from_items(rows).validate(**cast(Any, rules))

    assert len(pipeline.list_shards()) == 1
    with pytest.raises(ValidationError) as caught:
        pipeline.materialize()
    assert caught.value.rule == rule


def test_global_finalizer_holds_the_last_block_until_validation_passes() -> None:
    pipeline = (
        mdr.from_items([{"value": 1}]).with_max_block_rows(1).validate(exact_rows=2)
    )
    blocks = iter(pipeline.execute(pipeline.source.read()))

    with pytest.raises(ValidationError, match=r"\[exact_rows\]"):
        next(blocks)


def test_reusable_contract_and_inline_rules_are_mutually_exclusive() -> None:
    contract = ValidationContract(name="ids", unique=["id"], min_rows=1)
    pipeline = mdr.from_items([{"id": 1}]).validate(contract)

    assert _values(pipeline, "id") == [1]
    with pytest.raises(ValueError, match="either a ValidationContract or inline rules"):
        mdr.from_items([{"id": 1}]).validate(contract, not_null=["id"])


@pytest.mark.parametrize(
    "contract",
    [
        lambda: ValidationContract(),
        lambda: ValidationContract(exact_rows=1, min_rows=1),
        lambda: ValidationContract(min_rows=2, max_rows=1),
        lambda: ValidationContract(ranges={"x": (None, None)}),
        lambda: ValidationContract(unique_together=[("x",)]),
    ],
)
def test_contract_configuration_is_validated(contract) -> None:
    with pytest.raises(ValueError):
        contract()


def test_known_schema_rejects_missing_columns_before_execution() -> None:
    pipeline = RefinerPipeline(_TableSource(pa.table({"value": [1]})))

    with pytest.raises(ValueError, match="references missing column 'missing'"):
        pipeline.validate(not_null=["missing"])


@pytest.mark.parametrize("column", ["added", "value"])
def test_validation_recognizes_columns_assigned_by_with_column(column: str) -> None:
    pipeline = (
        RefinerPipeline(_TableSource(pa.table({"value": [1]})))
        .with_column(column, 2)
        .validate(not_null=[column], ranges={column: (1, 3)})
    )

    assert _values(pipeline, column) == [2]


def test_required_column_fails_for_empty_schema_less_input() -> None:
    pipeline = mdr.from_items([]).validate(not_null=["id"])

    with pytest.raises(ValidationError) as caught:
        pipeline.materialize()

    assert caught.value.rule == "column_exists:id"
    assert "empty input without a schema" in str(caught.value)


def test_known_schema_can_validate_after_filtering_every_row() -> None:
    pipeline = (
        RefinerPipeline(_TableSource(pa.table({"id": [1, 2]})))
        .filter(lambda row: False)
        .validate(not_null=["id"])
    )

    assert pipeline.materialize() == []


def test_global_contract_forces_one_worker_but_row_local_contract_does_not() -> None:
    global_pipeline = mdr.from_items([{"id": 1}]).validate(unique=["id"])
    local_pipeline = mdr.from_items([{"id": 1}]).validate(not_null=["id"])

    assert (
        plan_pipeline_stages(
            global_pipeline,
            default_num_workers=8,
        )[0].compute.num_workers
        == 1
    )
    assert (
        plan_pipeline_stages(
            local_pipeline,
            default_num_workers=8,
        )[0].compute.num_workers
        == 8
    )


def test_global_validation_claim_carries_the_exact_plan_to_workers() -> None:
    source = _TableSource(pa.table({"value": [1, 2]}))
    pipeline = RefinerPipeline(source).validate(unique=["value"])
    worker_payload = cloudpickle.dumps(pipeline)
    claimed_shard = Shard.from_dict(pipeline.list_shards()[0].to_dict())

    worker_pipeline = cloudpickle.loads(worker_payload)
    assert isinstance(claimed_shard.descriptor, ShardGroupDescriptor)
    rows = list(
        iter_rows(
            worker_pipeline.execute(
                worker_pipeline.source.iter_shard_units(claimed_shard)
            )
        )
    )
    assert [row["value"] for row in rows] == [1, 2]
    assert worker_pipeline.source.source.list_calls == 0


def test_validation_is_visible_in_compiled_plan_without_callable_payloads() -> None:
    pipeline = mdr.from_items([{"id": 1, "score": 0.5}]).validate(
        name="training_data",
        not_null=["id"],
        unique=["id"],
        ranges={"score": (0.0, 1.0)},
        predicates={"positive_id": lambda row: int(row["id"]) > 0},
    )

    stage = compile_pipeline_plan(pipeline)["stages"][0]
    step = stage["steps"][1]
    assert stage["requested_num_workers"] == 1
    assert step == {
        "name": "training_data",
        "type": "validation",
        "index": 1,
        "args": {
            "contract": "training_data",
            "not_null": ["id"],
            "unique": ["id"],
            "ranges": {"score": [0.0, 1.0]},
            "predicates": ["positive_id"],
            "scope": "global",
        },
    }


def test_validation_types_are_available_from_top_level_package() -> None:
    assert mdr.ValidationContract is ValidationContract
    assert mdr.ValidationError is ValidationError
