from __future__ import annotations

import builtins
from dataclasses import dataclass
from datetime import datetime as datetime_cls
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc


def _as_expr(value: Any) -> "Expr":
    if isinstance(value, Expr):
        return value
    return Expr(op="lit", args=(value,))


@dataclass(frozen=True, slots=True)
class Expr:
    op: builtins.str
    args: tuple[Any, ...]

    @property
    def str(self) -> "StringExpr":
        return StringExpr(self)

    @property
    def datetime(self) -> "DateTimeExpr":
        return DateTimeExpr(self)

    def is_null(self) -> "Expr":
        return Expr(op="is_null", args=(self,))

    def is_not_null(self) -> "Expr":
        return Expr(op="is_not_null", args=(self,))

    def is_in(self, values: list[Any] | tuple[Any, ...]) -> "Expr":
        return Expr(op="is_in", args=(self, tuple(values)))

    def between(self, lower: Any, upper: Any) -> "Expr":
        return (self >= _as_expr(lower)) & (self <= _as_expr(upper))

    def fill_null(self, value: Any) -> "Expr":
        return Expr(op="fill_null", args=(self, _as_expr(value)))

    def null_if(self, value: Any) -> "Expr":
        return Expr(op="null_if", args=(self, _as_expr(value)))

    def abs(self) -> "Expr":
        return Expr(op="abs", args=(self,))

    def floor(self) -> "Expr":
        return Expr(op="floor", args=(self,))

    def ceil(self) -> "Expr":
        return Expr(op="ceil", args=(self,))

    def round(self, ndigits: int = 0) -> "Expr":
        return Expr(op="round", args=(self, int(ndigits)))

    def clip(
        self, min_value: Any | None = None, max_value: Any | None = None
    ) -> "Expr":
        if min_value is None and max_value is None:
            raise ValueError("clip requires min_value and/or max_value")
        return Expr(
            op="clip",
            args=(
                self,
                _as_expr(min_value) if min_value is not None else None,
                _as_expr(max_value) if max_value is not None else None,
            ),
        )

    def to_plan(self) -> dict[builtins.str, Any]:
        def _serialize(v: Any) -> Any:
            if isinstance(v, Expr):
                return v.to_plan()
            if isinstance(v, (list, tuple)):
                return [_serialize(x) for x in v]
            if isinstance(v, dict):
                return {str(k): _serialize(val) for k, val in v.items()}
            if isinstance(v, datetime_cls):
                return v.isoformat()
            return v

        return {"op": self.op, "args": [_serialize(v) for v in self.args]}

    def __add__(self, other: Any) -> "Expr":
        return Expr(op="add", args=(self, _as_expr(other)))

    def __sub__(self, other: Any) -> "Expr":
        return Expr(op="sub", args=(self, _as_expr(other)))

    def __mul__(self, other: Any) -> "Expr":
        return Expr(op="mul", args=(self, _as_expr(other)))

    def __truediv__(self, other: Any) -> "Expr":
        return Expr(op="div", args=(self, _as_expr(other)))

    def __floordiv__(self, other: Any) -> "Expr":
        return Expr(op="floordiv", args=(self, _as_expr(other)))

    def __mod__(self, other: Any) -> "Expr":
        return Expr(op="mod", args=(self, _as_expr(other)))

    def __eq__(self, other: object) -> "Expr":  # type: ignore[override]
        return Expr(op="eq", args=(self, _as_expr(other)))

    def __ne__(self, other: object) -> "Expr":  # type: ignore[override]
        return Expr(op="ne", args=(self, _as_expr(other)))

    def __lt__(self, other: Any) -> "Expr":
        return Expr(op="lt", args=(self, _as_expr(other)))

    def __le__(self, other: Any) -> "Expr":
        return Expr(op="le", args=(self, _as_expr(other)))

    def __gt__(self, other: Any) -> "Expr":
        return Expr(op="gt", args=(self, _as_expr(other)))

    def __ge__(self, other: Any) -> "Expr":
        return Expr(op="ge", args=(self, _as_expr(other)))

    def __and__(self, other: Any) -> "Expr":
        return Expr(op="and", args=(self, _as_expr(other)))

    def __or__(self, other: Any) -> "Expr":
        return Expr(op="or", args=(self, _as_expr(other)))

    def __invert__(self) -> "Expr":
        return Expr(op="not", args=(self,))

    def __bool__(self) -> bool:
        raise TypeError(
            "Expr cannot be coerced to bool; use '&', '|' and '~' to compose predicates."
        )


@dataclass(frozen=True, slots=True)
class StringExpr:
    base: Expr

    def lower(self) -> Expr:
        return Expr(op="str_lower", args=(self.base,))

    def upper(self) -> Expr:
        return Expr(op="str_upper", args=(self.base,))

    def strip(self) -> Expr:
        return Expr(op="str_strip", args=(self.base,))

    def len(self) -> Expr:
        return Expr(op="str_len", args=(self.base,))

    def contains(self, pattern: str) -> Expr:
        return Expr(op="str_contains", args=(self.base, pattern))

    def startswith(self, prefix: str) -> Expr:
        return Expr(op="str_startswith", args=(self.base, prefix))

    def endswith(self, suffix: str) -> Expr:
        return Expr(op="str_endswith", args=(self.base, suffix))

    def regex_contains(self, pattern: str) -> Expr:
        return Expr(op="str_regex_contains", args=(self.base, pattern))

    def replace(self, pattern: str, replacement: str) -> Expr:
        return Expr(op="str_replace", args=(self.base, pattern, replacement))

    def regex_replace(self, pattern: str, replacement: str) -> Expr:
        return Expr(op="str_regex_replace", args=(self.base, pattern, replacement))


@dataclass(frozen=True, slots=True)
class DateTimeExpr:
    base: Expr

    def year(self) -> Expr:
        return Expr(op="datetime_year", args=(self.base,))

    def month(self) -> Expr:
        return Expr(op="datetime_month", args=(self.base,))

    def day(self) -> Expr:
        return Expr(op="datetime_day", args=(self.base,))

    def hour(self) -> Expr:
        return Expr(op="datetime_hour", args=(self.base,))

    def to_date(self) -> Expr:
        return Expr(op="datetime_to_date", args=(self.base,))


def col(name: str) -> Expr:
    return Expr(op="col", args=(name,))


def lit(value: Any) -> Expr:
    return Expr(op="lit", args=(value,))


def coalesce(*values: Any) -> Expr:
    return Expr(op="coalesce", args=tuple(_as_expr(v) for v in values))


def if_else(condition: Any, on_true: Any, on_false: Any) -> Expr:
    return Expr(
        op="if_else", args=(_as_expr(condition), _as_expr(on_true), _as_expr(on_false))
    )


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    return pc.call_function(name, list(args), **kwargs)


def _null_scalar_like(value: pa.Array | pa.ChunkedArray | pa.Scalar) -> pa.Scalar:
    return pa.scalar(None, type=value.type)


def eval_expr_arrow(
    expr: Expr, table: pa.Table | pa.RecordBatch
) -> pa.Array | pa.ChunkedArray | pa.Scalar:
    op = expr.op
    args = expr.args

    if op == "col":
        return table.column(str(args[0]))
    if op == "lit":
        return pa.scalar(args[0])
    if op == "coalesce":
        return _call("coalesce", *[eval_expr_arrow(v, table) for v in args])
    if op == "if_else":
        condition = eval_expr_arrow(args[0], table)
        on_true = eval_expr_arrow(args[1], table)
        on_false = eval_expr_arrow(args[2], table)
        return _call("if_else", condition, on_true, on_false)
    if op == "is_in":
        options = pc.SetLookupOptions(value_set=pa.array(list(args[1])))
        return _call("is_in", eval_expr_arrow(args[0], table), options=options)
    if op == "fill_null":
        return _call(
            "coalesce", eval_expr_arrow(args[0], table), eval_expr_arrow(args[1], table)
        )
    if op == "null_if":
        value = eval_expr_arrow(args[0], table)
        other = eval_expr_arrow(args[1], table)
        return _call(
            "if_else", _call("equal", value, other), _null_scalar_like(value), value
        )
    if op == "abs":
        return _call("abs", eval_expr_arrow(args[0], table))
    if op == "floor":
        return _call("floor", eval_expr_arrow(args[0], table))
    if op == "ceil":
        return _call("ceil", eval_expr_arrow(args[0], table))
    if op == "round":
        options = pc.RoundOptions(int(args[1]))
        return _call("round", eval_expr_arrow(args[0], table), options=options)
    if op == "clip":
        value = eval_expr_arrow(args[0], table)
        lower = args[1]
        upper = args[2]
        if lower is not None:
            lower_value = eval_expr_arrow(lower, table)
            value = _call(
                "if_else", _call("less", value, lower_value), lower_value, value
            )
        if upper is not None:
            upper_value = eval_expr_arrow(upper, table)
            value = _call(
                "if_else",
                _call("greater", value, upper_value),
                upper_value,
                value,
            )
        return value

    if op in {
        "add",
        "sub",
        "mul",
        "div",
        "floordiv",
        "mod",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "and",
        "or",
    }:
        left = eval_expr_arrow(args[0], table)
        right = eval_expr_arrow(args[1], table)
        binary_map = {
            "add": "add",
            "sub": "subtract",
            "mul": "multiply",
            "div": "divide",
            "eq": "equal",
            "ne": "not_equal",
            "lt": "less",
            "le": "less_equal",
            "gt": "greater",
            "ge": "greater_equal",
            "and": "and_kleene",
            "or": "or_kleene",
        }
        if op == "floordiv":
            return _call("floor", _call("divide", left, right))
        if op == "mod":
            floored = _call("floor", _call("divide", left, right))
            return _call("subtract", left, _call("multiply", floored, right))
        return _call(binary_map[op], left, right)

    if op == "not":
        return _call("invert", eval_expr_arrow(args[0], table))
    if op == "is_null":
        return _call("is_null", eval_expr_arrow(args[0], table))
    if op == "is_not_null":
        return _call("is_valid", eval_expr_arrow(args[0], table))

    if op == "str_lower":
        return _call("utf8_lower", eval_expr_arrow(args[0], table))
    if op == "str_upper":
        return _call("utf8_upper", eval_expr_arrow(args[0], table))
    if op == "str_strip":
        return _call("utf8_trim_whitespace", eval_expr_arrow(args[0], table))
    if op == "str_len":
        return _call("utf8_length", eval_expr_arrow(args[0], table))
    if op == "str_contains":
        options = pc.MatchSubstringOptions(pattern=str(args[1]))
        return _call(
            "match_substring", eval_expr_arrow(args[0], table), options=options
        )
    if op == "str_startswith":
        options = pc.MatchSubstringOptions(pattern=str(args[1]))
        return _call("starts_with", eval_expr_arrow(args[0], table), options=options)
    if op == "str_endswith":
        options = pc.MatchSubstringOptions(pattern=str(args[1]))
        return _call("ends_with", eval_expr_arrow(args[0], table), options=options)
    if op == "str_regex_contains":
        options = pc.MatchSubstringOptions(pattern=str(args[1]))
        return _call(
            "match_substring_regex",
            eval_expr_arrow(args[0], table),
            options=options,
        )
    if op == "str_replace":
        options = pc.ReplaceSubstringOptions(
            pattern=str(args[1]),
            replacement=str(args[2]),
        )
        return _call(
            "replace_substring", eval_expr_arrow(args[0], table), options=options
        )
    if op == "str_regex_replace":
        options = pc.ReplaceSubstringOptions(
            pattern=str(args[1]),
            replacement=str(args[2]),
        )
        return _call(
            "replace_substring_regex",
            eval_expr_arrow(args[0], table),
            options=options,
        )

    if op == "datetime_year":
        return _call("year", eval_expr_arrow(args[0], table))
    if op == "datetime_month":
        return _call("month", eval_expr_arrow(args[0], table))
    if op == "datetime_day":
        return _call("day", eval_expr_arrow(args[0], table))
    if op == "datetime_hour":
        return _call("hour", eval_expr_arrow(args[0], table))
    if op == "datetime_to_date":
        return pc.cast(eval_expr_arrow(args[0], table), target_type=pa.date32())

    raise ValueError(f"Unsupported expression op: {op}")


__all__ = [
    "Expr",
    "StringExpr",
    "DateTimeExpr",
    "col",
    "lit",
    "coalesce",
    "if_else",
    "eval_expr_arrow",
]
