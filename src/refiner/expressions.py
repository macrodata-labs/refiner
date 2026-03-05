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

    def replace(self, pattern: str, replacement: str) -> Expr:
        return Expr(op="str_replace", args=(self.base, pattern, replacement))


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


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    return pc.call_function(name, list(args), **kwargs)


def eval_expr_arrow(
    expr: Expr, table: pa.Table
) -> pa.Array | pa.ChunkedArray | pa.Scalar:
    op = expr.op
    args = expr.args

    if op == "col":
        return table.column(str(args[0]))
    if op == "lit":
        return pa.scalar(args[0])
    if op == "coalesce":
        return _call("coalesce", *[eval_expr_arrow(v, table) for v in args])

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
        return _call(
            "match_substring",
            eval_expr_arrow(args[0], table),
            pattern=str(args[1]),
        )
    if op == "str_replace":
        return _call(
            "replace_substring",
            eval_expr_arrow(args[0], table),
            pattern=str(args[1]),
            replacement=str(args[2]),
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
    "eval_expr_arrow",
]
