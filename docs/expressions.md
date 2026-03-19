---
title: "Expressions"
description: "The expression DSL used by Refiner's vectorized transforms"
---

Refiner expressions are the small DSL used by vectorized pipeline methods like `.filter(...)`, `.with_columns(...)`, and `.select(...)`.

Expressions are evaluated column-wise on Arrow data, not row-by-row in Python.

## Core constructors

| signature | what it does |
| --- | --- |
| `mdr.col(name: str)` | references an existing column by name |
| `mdr.lit(value: Any)` | wraps a Python literal so it can be used inside an expression |
| `mdr.coalesce(*values: Any)` | returns the first non-null value among its arguments |
| `mdr.if_else(condition: Any, on_true: Any, on_false: Any)` | chooses between two values based on a boolean condition |

Example:

```python
import refiner as mdr

expr = mdr.if_else(
    mdr.col("lang") == "en",
    mdr.col("text"),
    mdr.lit(None),
)
```

## Boolean composition

Use bitwise operators to combine predicates:

| syntax | what it means |
| --- | --- |
| `left & right` | boolean AND |
| `left | right` | boolean OR |
| `~expr` | boolean NOT |

Example:

```python
predicate = (
    (mdr.col("lang") == "en")
    & (mdr.col("text").is_not_null())
    & (mdr.col("text").str.strip() != "")
)
```

Do not use Python `and`, `or`, or `not` with expressions. Those operators try to coerce the expression to a Python boolean and will fail.

## Common expression methods

| signature | what it does |
| --- | --- |
| `expr.is_null()` | true where the value is null |
| `expr.is_not_null()` | true where the value is present |
| `expr.is_in(values)` | true where the scalar value appears in the provided set, or where a list-valued column contains any element from that set |
| `expr.between(lower, upper)` | true where the value is between the two bounds |
| `expr.fill_null(value)` | replaces nulls with the provided fallback |
| `expr.null_if(value)` | turns matching values into null |
| `expr.abs()` | absolute value |
| `expr.floor()` | floor for numeric expressions |
| `expr.ceil()` | ceiling for numeric expressions |
| `expr.round(ndigits=0)` | rounds numeric expressions |
| `expr.clip(min_value=None, max_value=None)` | clamps values to the provided lower and/or upper bound |

Expressions also support arithmetic and comparisons directly:

```python
score = (mdr.col("correct") / mdr.col("total")).clip(min_value=0, max_value=1)
```

For list-valued columns, `.is_in(...)` is a membership check against the list
contents, not whole-list equality:

```python
predicate = mdr.col("tasks").is_in(["pick", "place"])
```

That means the predicate is true if an episode's `tasks` list contains either
`"pick"` or `"place"`.

## String namespace

Access string methods through `.str`:

| signature | what it does |
| --- | --- |
| `expr.str.lower()` | lowercases strings |
| `expr.str.upper()` | uppercases strings |
| `expr.str.strip()` | trims leading and trailing whitespace |
| `expr.str.len()` | returns string length |
| `expr.str.contains(pattern: str)` | true where the string contains a substring |
| `expr.str.startswith(prefix: str)` | true where the string starts with the prefix |
| `expr.str.endswith(suffix: str)` | true where the string ends with the suffix |
| `expr.str.regex_contains(pattern: str)` | true where the regex matches somewhere in the string |
| `expr.str.replace(pattern: str, replacement: str)` | replaces substring matches |
| `expr.str.regex_replace(pattern: str, replacement: str)` | replaces regex matches |

Example:

```python
clean_text = mdr.col("text").str.strip().str.lower()
```

## Datetime namespace

Access datetime methods through `.datetime`:

| signature | what it does |
| --- | --- |
| `expr.datetime.year()` | extracts the year |
| `expr.datetime.month()` | extracts the month |
| `expr.datetime.day()` | extracts the day |
| `expr.datetime.hour()` | extracts the hour |
| `expr.datetime.to_date()` | truncates a datetime to a date |

## Where expressions are used

Expressions are accepted by:

- `.filter(expr)`
- `.with_columns(...)`
- `.with_column(name, expr)`

They also participate inside helpers like:

- `mdr.coalesce(...)`
- `mdr.if_else(...)`

## Related pages

- [Transforms](transforms.md)
- [Reading and writing data](reading-and-writing.md)
- [Pipeline basics](pipeline-basics.md)
