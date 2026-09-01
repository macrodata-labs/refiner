from __future__ import annotations

import math
import pickle
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NoReturn, TypeAlias, cast

import pyarrow as pa
import pyarrow.compute as pc

from refiner.pipeline.data.block import Block
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.shard import SHARD_ID_COLUMN, SOURCE_ROW_ID_COLUMN
from refiner.pipeline.data.tabular import Tabular

ValidationPredicate: TypeAlias = Callable[[Row], bool]
RangeBounds: TypeAlias = tuple[Any | None, Any | None]


@dataclass(frozen=True, slots=True)
class RowLocation:
    shard_id: str | None
    source_row_id: int | None

    def describe(self) -> str:
        details: list[str] = []
        if self.shard_id is not None:
            details.append(f"shard_id={self.shard_id!r}")
        if self.source_row_id is not None:
            details.append(f"source_row_id={self.source_row_id}")
        return " ".join(details)


class ValidationError(ValueError):
    """Raised when a pipeline row violates a validation contract."""

    def __init__(
        self,
        *,
        contract_name: str,
        rule: str,
        detail: str,
        location: RowLocation | None = None,
    ) -> None:
        self.contract_name = contract_name
        self.rule = rule
        self.detail = detail
        self.location = location
        location_text = location.describe() if location is not None else ""
        suffix = f" ({location_text})" if location_text else ""
        super().__init__(
            f"Validation {contract_name!r} failed [{rule}]: {detail}{suffix}"
        )


def _normalized_columns(values: Sequence[str], *, argument: str) -> tuple[str, ...]:
    columns = (values,) if isinstance(values, str) else tuple(values)
    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError(f"{argument} must contain non-empty column names")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{argument} must not contain duplicate column names")
    return columns


@dataclass(frozen=True, slots=True)
class ValidationContract:
    """Reusable constraints for one point in a pipeline."""

    name: str = "validation"
    not_null: Sequence[str] = ()
    unique: Sequence[str] = ()
    unique_together: Sequence[Sequence[str]] = ()
    ranges: Mapping[str, RangeBounds] = field(default_factory=dict)
    min_rows: int | None = None
    max_rows: int | None = None
    exact_rows: int | None = None
    predicates: Mapping[str, ValidationPredicate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("validation contract name must be non-empty")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(
            self,
            "not_null",
            _normalized_columns(self.not_null, argument="not_null"),
        )
        object.__setattr__(
            self,
            "unique",
            _normalized_columns(self.unique, argument="unique"),
        )

        unique_together: list[tuple[str, ...]] = []
        for columns in self.unique_together:
            normalized = _normalized_columns(
                columns,
                argument="unique_together",
            )
            if len(normalized) < 2:
                raise ValueError(
                    "unique_together entries must contain at least two columns"
                )
            unique_together.append(normalized)
        if len(set(unique_together)) != len(unique_together):
            raise ValueError("unique_together must not contain duplicate constraints")
        object.__setattr__(self, "unique_together", tuple(unique_together))

        ranges: dict[str, RangeBounds] = {}
        for column, bounds in self.ranges.items():
            if not isinstance(column, str) or not column:
                raise ValueError("ranges keys must be non-empty column names")
            if not isinstance(bounds, tuple | list) or len(bounds) != 2:
                raise ValueError(
                    f"range for {column!r} must be a (minimum, maximum) pair"
                )
            lower, upper = bounds
            if lower is None and upper is None:
                raise ValueError(f"range for {column!r} must include a bound")
            if lower is not None and upper is not None:
                try:
                    invalid_order = lower > upper
                except Exception as err:
                    raise ValueError(
                        f"range bounds for {column!r} are not comparable"
                    ) from err
                if invalid_order:
                    raise ValueError(
                        f"range minimum for {column!r} must be <= its maximum"
                    )
            ranges[column] = (lower, upper)
        object.__setattr__(self, "ranges", ranges)

        for argument, value in (
            ("min_rows", self.min_rows),
            ("max_rows", self.max_rows),
            ("exact_rows", self.exact_rows),
        ):
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{argument} must be a non-negative integer")
        if self.exact_rows is not None and (
            self.min_rows is not None or self.max_rows is not None
        ):
            raise ValueError("exact_rows cannot be combined with min_rows or max_rows")
        if (
            self.min_rows is not None
            and self.max_rows is not None
            and self.min_rows > self.max_rows
        ):
            raise ValueError("min_rows must be <= max_rows")

        predicates = dict(self.predicates)
        for predicate_name, predicate in predicates.items():
            if not isinstance(predicate_name, str) or not predicate_name:
                raise ValueError("predicate names must be non-empty strings")
            if not callable(predicate):
                raise TypeError(f"predicate {predicate_name!r} must be callable")
        object.__setattr__(self, "predicates", predicates)

        if not any(
            (
                self.not_null,
                self.unique,
                self.unique_together,
                self.ranges,
                self.min_rows is not None,
                self.max_rows is not None,
                self.exact_rows is not None,
                self.predicates,
            )
        ):
            raise ValueError("validation contract must contain at least one rule")

    @property
    def requires_global_scope(self) -> bool:
        return bool(
            self.unique
            or self.unique_together
            or self.min_rows is not None
            or self.max_rows is not None
            or self.exact_rows is not None
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns = list(self.not_null)
        columns.extend(self.unique)
        for group in self.unique_together:
            columns.extend(group)
        columns.extend(self.ranges)
        return tuple(dict.fromkeys(columns))

    def validate_schema(self, schema: pa.Schema | None) -> None:
        self.validate_columns(None if schema is None else schema.names)

    def validate_columns(self, columns: Sequence[str] | None) -> None:
        if columns is None:
            return
        missing = [column for column in self.required_columns if column not in columns]
        if missing:
            raise ValueError(
                f"Validation contract {self.name!r} references missing column "
                f"{missing[0]!r}"
            )

    def describe(self) -> dict[str, Any]:
        description: dict[str, Any] = {"contract": self.name}
        if self.not_null:
            description["not_null"] = list(self.not_null)
        if self.unique:
            description["unique"] = list(self.unique)
        if self.unique_together:
            description["unique_together"] = [
                list(columns) for columns in self.unique_together
            ]
        if self.ranges:
            description["ranges"] = {
                column: [_display_value(lower), _display_value(upper)]
                for column, (lower, upper) in self.ranges.items()
            }
        if self.min_rows is not None:
            description["min_rows"] = self.min_rows
        if self.max_rows is not None:
            description["max_rows"] = self.max_rows
        if self.exact_rows is not None:
            description["exact_rows"] = self.exact_rows
        if self.predicates:
            description["predicates"] = list(self.predicates)
        description["scope"] = "global" if self.requires_global_scope else "row_local"
        return description


class ValidationRuntime:
    """Mutable state for one execution of an immutable contract."""

    def __init__(
        self,
        contract: ValidationContract,
        *,
        known_columns: Sequence[str] | None = None,
    ) -> None:
        self.contract = contract
        self.known_columns = None if known_columns is None else tuple(known_columns)
        self.row_count = 0
        self._columns_observed = False
        self._seen: dict[str, dict[Any, RowLocation]] = {
            f"unique:{column}": {} for column in contract.unique
        }
        self._seen.update(
            {
                f"unique_together:{','.join(columns)}": {}
                for columns in contract.unique_together
            }
        )

    def validate_block(self, block: Block) -> None:
        block_rows = block.num_rows if isinstance(block, Tabular) else len(block)
        previous_count = self.row_count
        self.row_count += block_rows
        if (
            self.contract.max_rows is not None
            and self.row_count > self.contract.max_rows
        ):
            index = max(0, self.contract.max_rows - previous_count)
            self._fail(
                rule="max_rows",
                detail=(
                    f"expected at most {self.contract.max_rows} rows, observed at "
                    f"least {self.contract.max_rows + 1}"
                ),
                location=_block_location(block, index),
            )

        if isinstance(block, Tabular):
            self._validate_table(block)
        else:
            self._validate_rows(block)

    def finalize(self) -> None:
        if (
            self.contract.exact_rows is not None
            and self.row_count != self.contract.exact_rows
        ):
            self._fail(
                rule="exact_rows",
                detail=(
                    f"expected exactly {self.contract.exact_rows} rows, "
                    f"observed {self.row_count}"
                ),
            )
        if (
            self.contract.min_rows is not None
            and self.row_count < self.contract.min_rows
        ):
            self._fail(
                rule="min_rows",
                detail=(
                    f"expected at least {self.contract.min_rows} rows, "
                    f"observed {self.row_count}"
                ),
            )
        if self.contract.required_columns and not self._columns_observed:
            if self.known_columns is None:
                column = self.contract.required_columns[0]
                self._fail(
                    rule=f"column_exists:{column}",
                    detail=(
                        f"required column {column!r} cannot be established from "
                        "an empty input without a schema"
                    ),
                )
            self._require_columns(self.known_columns)

    def _validate_table(self, block: Tabular) -> None:
        table = block.table
        self._require_columns(table.column_names)

        for column in self.contract.not_null:
            values = table.column(column)
            if values.null_count:
                index = _first_true(_call_compute("is_null", values))
                self._fail(
                    rule=f"not_null:{column}",
                    detail=f"column {column!r} contains null",
                    location=_table_location(table, index),
                )

        for column, bounds in self.contract.ranges.items():
            values = table.column(column)
            try:
                invalid = _outside_range(values, bounds, column=column)
            except TypeError:
                self._validate_range_values(
                    column=column,
                    bounds=bounds,
                    values=values.to_pylist(),
                    locations=(
                        _table_location(table, index) for index in range(table.num_rows)
                    ),
                )
                continue
            index = _first_true(invalid)
            if index is not None:
                self._fail(
                    rule=f"range:{column}",
                    detail=(
                        f"column {column!r} value {_short_repr(values[index].as_py())} "
                        f"is outside {_range_text(bounds)}"
                    ),
                    location=_table_location(table, index),
                )

        for column in self.contract.unique:
            values = table.column(column).to_pylist()
            for index, value in enumerate(values):
                self._check_unique(
                    rule=f"unique:{column}",
                    key=value,
                    display=value,
                    location=_table_location(table, index),
                )

        for columns in self.contract.unique_together:
            values_by_column = [table.column(column).to_pylist() for column in columns]
            rule = f"unique_together:{','.join(columns)}"
            for index, values in enumerate(zip(*values_by_column, strict=True)):
                self._check_unique(
                    rule=rule,
                    key=values,
                    display=values,
                    location=_table_location(table, index),
                )

        if self.contract.predicates:
            self._validate_predicates(block.to_rows())

    def _validate_rows(self, rows: Sequence[Row]) -> None:
        for row in rows:
            self._require_columns(tuple(row.keys()))
            location = RowLocation(row.shard_id, row.source_row_id)
            for column in self.contract.not_null:
                if row[column] is None:
                    self._fail(
                        rule=f"not_null:{column}",
                        detail=f"column {column!r} contains null",
                        location=location,
                    )
            for column, bounds in self.contract.ranges.items():
                value = row[column]
                self._validate_range_values(
                    column=column,
                    bounds=bounds,
                    values=(value,),
                    locations=(location,),
                )
            for column in self.contract.unique:
                value = row[column]
                self._check_unique(
                    rule=f"unique:{column}",
                    key=value,
                    display=value,
                    location=location,
                )
            for columns in self.contract.unique_together:
                values = tuple(row[column] for column in columns)
                self._check_unique(
                    rule=f"unique_together:{','.join(columns)}",
                    key=values,
                    display=values,
                    location=location,
                )
        if self.contract.predicates:
            self._validate_predicates(rows)

    def _validate_predicates(self, rows: Sequence[Row]) -> None:
        for row in rows:
            location = RowLocation(row.shard_id, row.source_row_id)
            for name, predicate in self.contract.predicates.items():
                try:
                    valid = bool(predicate(row))
                except Exception as err:
                    validation_error = ValidationError(
                        contract_name=self.contract.name,
                        rule=f"predicate:{name}",
                        detail=(
                            f"predicate raised {type(err).__name__}: "
                            f"{str(err).strip() or type(err).__name__}"
                        ),
                        location=location,
                    )
                    raise validation_error from err
                if not valid:
                    self._fail(
                        rule=f"predicate:{name}",
                        detail=f"predicate {name!r} returned false",
                        location=location,
                    )

    def _validate_range_values(
        self,
        *,
        column: str,
        bounds: RangeBounds,
        values: Sequence[Any],
        locations: Iterable[RowLocation],
    ) -> None:
        for value, location in zip(values, locations, strict=True):
            if value is None:
                continue
            try:
                valid = _value_in_range(value, bounds)
            except TypeError as err:
                validation_error = ValidationError(
                    contract_name=self.contract.name,
                    rule=f"range:{column}",
                    detail=str(err),
                    location=location,
                )
                raise validation_error from err
            if not valid:
                self._fail(
                    rule=f"range:{column}",
                    detail=(
                        f"column {column!r} value {_short_repr(value)} is outside "
                        f"{_range_text(bounds)}"
                    ),
                    location=location,
                )

    def _require_columns(self, available: Sequence[str]) -> None:
        available_set = set(available)
        for column in self.contract.required_columns:
            if column not in available_set:
                self._fail(
                    rule=f"column_exists:{column}",
                    detail=f"required column {column!r} is missing",
                )
        self._columns_observed = True

    def _check_unique(
        self,
        *,
        rule: str,
        key: Any,
        display: Any,
        location: RowLocation,
    ) -> None:
        frozen = _freeze_value(key)
        seen = self._seen[rule]
        first_location = seen.get(frozen)
        if first_location is None:
            seen[frozen] = location
            return
        first_text = first_location.describe() or "an earlier row"
        self._fail(
            rule=rule,
            detail=(
                f"duplicate value {_short_repr(display)}; first observed at "
                f"{first_text}"
            ),
            location=location,
        )

    def _fail(
        self,
        *,
        rule: str,
        detail: str,
        location: RowLocation | None = None,
    ) -> NoReturn:
        raise ValidationError(
            contract_name=self.contract.name,
            rule=rule,
            detail=detail,
            location=location,
        )


def _display_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return _short_repr(value)


def _short_repr(value: Any, *, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _range_text(bounds: RangeBounds) -> str:
    lower, upper = bounds
    if lower is None:
        return f"(-inf, {_short_repr(upper)}]"
    if upper is None:
        return f"[{_short_repr(lower)}, +inf)"
    return f"[{_short_repr(lower)}, {_short_repr(upper)}]"


def _value_in_range(value: Any, bounds: RangeBounds) -> bool:
    lower, upper = bounds
    try:
        if _is_nan(value):
            return False
        return not (
            (lower is not None and value < lower)
            or (upper is not None and value > upper)
        )
    except Exception as err:
        raise TypeError(
            f"value {_short_repr(value)} cannot be compared with range "
            f"{_range_text(bounds)}"
        ) from err


def _outside_range(
    values: pa.ChunkedArray,
    bounds: RangeBounds,
    *,
    column: str,
) -> pa.Array | pa.ChunkedArray:
    lower, upper = bounds
    try:
        valid: pa.Array | pa.ChunkedArray | None = None
        if lower is not None:
            valid = _call_compute("greater_equal", values, pa.scalar(lower))
        if upper is not None:
            upper_valid = _call_compute("less_equal", values, pa.scalar(upper))
            valid = (
                upper_valid
                if valid is None
                else _call_compute("and_kleene", valid, upper_valid)
            )
        assert valid is not None
        return _call_compute("invert", pc.fill_null(valid, True))
    except Exception as err:
        raise TypeError(
            f"column {column!r} cannot be compared with range {_range_text(bounds)}"
        ) from err


def _first_true(values: pa.Array | pa.ChunkedArray) -> int | None:
    indices = _call_compute("indices_nonzero", values)
    return None if len(indices) == 0 else int(indices[0].as_py())


def _call_compute(
    function: str,
    *arguments: pa.Array | pa.ChunkedArray | pa.Scalar,
) -> pa.Array | pa.ChunkedArray:
    return cast(
        pa.Array | pa.ChunkedArray,
        pc.call_function(function, list(arguments)),
    )


def _block_location(block: Block, index: int) -> RowLocation | None:
    if isinstance(block, Tabular):
        return _table_location(block.table, index)
    if 0 <= index < len(block):
        row = block[index]
        return RowLocation(row.shard_id, row.source_row_id)
    return None


def _table_location(table: pa.Table, index: int | None) -> RowLocation:
    if index is None or index < 0 or index >= table.num_rows:
        return RowLocation(None, None)
    shard_id = (
        table.column(SHARD_ID_COLUMN)[index].as_py()
        if SHARD_ID_COLUMN in table.column_names
        else None
    )
    source_row_id = (
        table.column(SOURCE_ROW_ID_COLUMN)[index].as_py()
        if SOURCE_ROW_ID_COLUMN in table.column_names
        else None
    )
    return RowLocation(
        str(shard_id) if shard_id is not None else None,
        int(source_row_id) if source_row_id is not None else None,
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, pa.Scalar):
        value = value.as_py()
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                sorted(
                    (
                        (_freeze_value(key), _freeze_value(item))
                        for key, item in value.items()
                    ),
                    key=repr,
                )
            ),
        )
    if isinstance(value, list):
        return ("list", tuple(_freeze_value(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze_value(item) for item in value))
    if isinstance(value, set | frozenset):
        return (
            "set",
            tuple(sorted((_freeze_value(item) for item in value), key=repr)),
        )
    if _is_nan(value):
        return ("float", "nan")
    try:
        hash(value)
    except Exception:
        try:
            return (
                type(value).__module__,
                type(value).__qualname__,
                pickle.dumps(value),
            )
        except Exception:
            return (type(value).__module__, type(value).__qualname__, repr(value))
    return ("hashable", value)


def _is_nan(value: Any) -> bool:
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "RangeBounds",
    "ValidationContract",
    "ValidationError",
    "ValidationPredicate",
]
