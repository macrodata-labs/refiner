---
title: "Expression Transforms"
description: "Use expression-backed pipeline transforms that run on the vectorized Arrow path"
---

Refiner supports expression-backed transforms that keep the same pipeline style while enabling a fast vectorized execution path.

## Build Expressions

```python
import refiner as mdr

expr = (mdr.col("x") + 1) * 2
```

Core constructors:

- `mdr.col("name")`
- `mdr.lit(value)`
- `mdr.coalesce(...)`
- `mdr.if_else(condition, on_true, on_false)`

Use `&`, `|`, and `~` to combine boolean expressions. Do not use Python `and`, `or`, or `not` with `Expr` objects.

## Vectorized Transforms

These methods are shard-local and expression-backed:

- `.select(*cols)`
- `.with_columns(**exprs)`
- `.with_column(name, expr)`
- `.drop(*cols)`
- `.rename(**mapping)`
- `.cast(**dtype_map)`
- `.filter(expr)`

Example:

```python
import refiner as mdr

pipeline = (
    mdr.read_parquet("data/*.parquet")
    .filter(mdr.col("lang") == "en")
    .with_columns(
        text_len=mdr.col("text").str.len(),
        day=mdr.col("timestamp").datetime.to_date(),
    )
    .select("doc_id", "text_len", "day")
)
```

## String and Datetime Namespaces

String namespace (`.str`):

- `.lower()`, `.upper()`, `.strip()`, `.len()`
- `.contains(pattern)`, `.startswith(prefix)`, `.endswith(suffix)`
- `.regex_contains(pattern)`, `.replace(pattern, replacement)`, `.regex_replace(pattern, replacement)`

Datetime namespace (`.datetime`):

- `.year()`, `.month()`, `.day()`, `.hour()`, `.to_date()`

General expression helpers:

- Null and conditional: `.is_null()`, `.is_not_null()`, `.fill_null(value)`, `.null_if(value)`, `mdr.if_else(...)`
- Membership and ranges: `.is_in(values)`, `.between(lower, upper)`
- Numeric unary: `.abs()`, `.floor()`, `.ceil()`, `.round(ndigits=0)`, `.clip(min_value=..., max_value=...)`

Example:

```python
pipeline = (
    mdr.from_items(
        [
            {"x": 1, "text": "foo1", "score": None, "z": 1.2},
            {"x": 2, "text": "bar2", "score": 5, "z": 2.6},
        ]
    )
    .with_columns(
        keep=mdr.col("x").is_in([1, 3]),
        x_band=mdr.col("x").between(1, 2),
        prefix=mdr.col("text").str.startswith("ba"),
        has_digit=mdr.col("text").str.regex_contains(r"\d"),
        text_clean=mdr.col("text").str.regex_replace(r"\d", ""),
        score=mdr.col("score").fill_null(0),
        score_bucket=mdr.if_else(mdr.col("score") > 3, "high", "low"),
        z=mdr.col("z").clip(min_value=1.5, max_value=2.5).round(0),
    )
)
```

## Interop with Python UDF Steps

Expression-backed transforms and Python UDF steps can be mixed in one pipeline.

```python
pipeline = (
    mdr.from_items([{"x": 1}, {"x": 2}])
    .with_columns(y=mdr.col("x") + 1)
    .map(lambda row: {"z": int(row["y"]) * 10})
    .with_column("w", mdr.col("z") + 5)
)
```

Refiner handles boundaries internally: expression runs on Arrow blocks, UDF runs on row mode.

## Internal Notes

- Adjacent expression-backed operations are fused into one vectorized segment.
- Vectorized segments convert row input to Arrow once per segment and convert back once.
- Global operators like `sort`, `limit`, `distinct`, `groupby`, and `join` are intentionally out of scope in this phase.
