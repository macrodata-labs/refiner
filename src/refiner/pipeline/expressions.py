from __future__ import annotations

import builtins
from dataclasses import dataclass
from datetime import datetime as datetime_cls
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

_ARROW_FUNCTIONS = frozenset(pc.list_functions())
_HAS_FLOOR_DIVIDE_KERNEL = "floor_divide" in _ARROW_FUNCTIONS
_HAS_MOD_KERNEL = "mod" in _ARROW_FUNCTIONS
_HAS_MAXIMUM_KERNEL = "maximum" in _ARROW_FUNCTIONS
_HAS_MINIMUM_KERNEL = "minimum" in _ARROW_FUNCTIONS
_ELEMENT_WISE_KEEP_NULLS = pc.ElementWiseAggregateOptions(skip_nulls=False)


def _as_expr(value: Any) -> "Expr":
    if isinstance(value, Expr):
        return value
    return Expr(op="lit", args=(value,))


def value_to_code(value: Any) -> str:
    if isinstance(value, Expr):
        return expr_to_code(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(value_to_code(item) for item in value) + "]"
    if isinstance(value, tuple):
        if len(value) == 1:
            return f"({value_to_code(value[0])},)"
        return "(" + ", ".join(value_to_code(item) for item in value) + ")"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{value_to_code(key)}: {value_to_code(item)}"
                for key, item in value.items()
            )
            + "}"
        )
    return repr(value)


def expr_to_code(value: Any) -> str:
    if not isinstance(value, Expr):
        return value_to_code(value)

    op = value.op
    args = value.args

    if op == "col":
        return f"col({args[0]!r})"
    if op == "lit":
        return value_to_code(args[0])
    if op == "coalesce":
        return f"coalesce({', '.join(expr_to_code(arg) for arg in args)})"
    if op == "if_else":
        return f"if_else({expr_to_code(args[0])}, {expr_to_code(args[1])}, {expr_to_code(args[2])})"

    binary_ops = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "floordiv": "//",
        "mod": "%",
        "eq": "==",
        "ne": "!=",
        "lt": "<",
        "le": "<=",
        "gt": ">",
        "ge": ">=",
        "and": "&",
        "or": "|",
    }
    if op in binary_ops:
        return f"({expr_to_code(args[0])} {binary_ops[op]} {expr_to_code(args[1])})"
    if op == "not":
        return f"(~{expr_to_code(args[0])})"

    if op == "is_null":
        return f"{expr_to_code(args[0])}.is_null()"
    if op == "is_not_null":
        return f"{expr_to_code(args[0])}.is_not_null()"
    if op == "is_in":
        return f"{expr_to_code(args[0])}.is_in({value_to_code(list(args[1]))})"
    if op == "fill_null":
        return f"{expr_to_code(args[0])}.fill_null({expr_to_code(args[1])})"
    if op == "null_if":
        return f"{expr_to_code(args[0])}.null_if({expr_to_code(args[1])})"

    if op == "abs":
        return f"{expr_to_code(args[0])}.abs()"
    if op == "floor":
        return f"{expr_to_code(args[0])}.floor()"
    if op == "ceil":
        return f"{expr_to_code(args[0])}.ceil()"
    if op == "round":
        return f"{expr_to_code(args[0])}.round({value_to_code(args[1])})"
    if op == "clip":
        kwargs: list[str] = []
        if args[1] is not None:
            kwargs.append(f"min_value={expr_to_code(args[1])}")
        if args[2] is not None:
            kwargs.append(f"max_value={expr_to_code(args[2])}")
        return f"{expr_to_code(args[0])}.clip({', '.join(kwargs)})"

    if op == "str_lower":
        return f"{expr_to_code(args[0])}.str.lower()"
    if op == "str_upper":
        return f"{expr_to_code(args[0])}.str.upper()"
    if op == "str_strip":
        return f"{expr_to_code(args[0])}.str.strip()"
    if op == "str_len":
        return f"{expr_to_code(args[0])}.str.len()"
    if op == "str_contains":
        return f"{expr_to_code(args[0])}.str.contains({value_to_code(args[1])})"
    if op == "str_startswith":
        return f"{expr_to_code(args[0])}.str.startswith({value_to_code(args[1])})"
    if op == "str_endswith":
        return f"{expr_to_code(args[0])}.str.endswith({value_to_code(args[1])})"
    if op == "str_regex_contains":
        return f"{expr_to_code(args[0])}.str.regex_contains({value_to_code(args[1])})"
    if op == "str_replace":
        return (
            f"{expr_to_code(args[0])}.str.replace("
            f"{value_to_code(args[1])}, {value_to_code(args[2])})"
        )
    if op == "str_regex_replace":
        return (
            f"{expr_to_code(args[0])}.str.regex_replace("
            f"{value_to_code(args[1])}, {value_to_code(args[2])})"
        )

    if op == "datetime_year":
        return f"{expr_to_code(args[0])}.datetime.year()"
    if op == "datetime_month":
        return f"{expr_to_code(args[0])}.datetime.month()"
    if op == "datetime_day":
        return f"{expr_to_code(args[0])}.datetime.day()"
    if op == "datetime_hour":
        return f"{expr_to_code(args[0])}.datetime.hour()"
    if op == "datetime_to_date":
        return f"{expr_to_code(args[0])}.datetime.to_date()"

    return value_to_code(value.to_plan())


def with_columns_assignments_to_code(assignments: dict[str, Any]) -> str:
    return ", ".join(
        f"{name}={expr.to_code() if isinstance(expr, Expr) else expr_to_code(expr)}"
        for name, expr in assignments.items()
    )


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

    def to_code(self) -> builtins.str:
        return expr_to_code(self)

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
        value = eval_expr_arrow(args[0], table)
        if (
            pa.types.is_list(value.type)
            or pa.types.is_large_list(value.type)
            or pa.types.is_fixed_size_list(value.type)
        ):
            lists = (
                value.combine_chunks() if isinstance(value, pa.ChunkedArray) else value
            )
            flattened = _call("list_flatten", lists)
            if len(flattened) == 0:
                return pa.array([False] * len(lists), type=pa.bool_())
            parent_indices = _call("list_parent_indices", lists)
            matches = _call("is_in", flattened, options=options)
            grouped = (
                pa.table({"parent": parent_indices, "match": matches})
                .group_by("parent")
                .aggregate([("match", "any")])
            )
            result = [False] * len(lists)
            for parent, matched in zip(
                grouped.column("parent").to_pylist(),
                grouped.column("match_any").to_pylist(),
                strict=True,
            ):
                result[int(parent)] = bool(matched)
            return pa.array(result, type=pa.bool_())
        return _call("is_in", value, options=options)
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
            if _HAS_MAXIMUM_KERNEL:
                value = _call("maximum", value, lower_value)
            else:
                # `maximum` is not available on older Arrow versions.
                value = _call(
                    "max_element_wise",
                    value,
                    lower_value,
                    options=_ELEMENT_WISE_KEEP_NULLS,
                )
        if upper is not None:
            upper_value = eval_expr_arrow(upper, table)
            if _HAS_MINIMUM_KERNEL:
                value = _call("minimum", value, upper_value)
            else:
                # `minimum` is not available on older Arrow versions.
                value = _call(
                    "min_element_wise",
                    value,
                    upper_value,
                    options=_ELEMENT_WISE_KEEP_NULLS,
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
            "floordiv": "floor_divide" if _HAS_FLOOR_DIVIDE_KERNEL else None,
            "mod": "mod" if _HAS_MOD_KERNEL else None,
            "eq": "equal",
            "ne": "not_equal",
            "lt": "less",
            "le": "less_equal",
            "gt": "greater",
            "ge": "greater_equal",
            "and": "and_kleene",
            "or": "or_kleene",
        }
        direct_kernel = binary_map[op]
        if direct_kernel is not None:
            return _call(direct_kernel, left, right)
        if op == "floordiv":
            return _call("floor", _call("divide", left, right))
        if op == "mod":
            floored = _call("floor", _call("divide", left, right))
            return _call("subtract", left, _call("multiply", floored, right))
        raise ValueError(f"Unsupported binary expression op: {op}")

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
