---
title: "Expression Transforms"
description: "Use Arrow-backed expression transforms in Refiner pipelines"
---

Refiner supports expression-backed transforms that run on the vectorized Arrow path.

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

Use:

- `&`
- `|`
- `~`

for boolean composition. Do not use Python `and`, `or`, or `not` with `Expr` objects.

## Pipeline methods

These methods operate on the current row schema and stay shard-local.

| method | what it does | input shape | output shape |
| --- | --- | --- | --- |
| `.filter(expr)` | keeps only rows where the expression is truthy | any rows with the referenced columns present | same columns, fewer rows |
| `.with_columns(...)` | adds or overwrites multiple columns | any rows with the referenced columns present | same rows, wider schema |
| `.with_column(...)` | adds or overwrites one column | any rows with the referenced columns present | same rows, wider schema |
| `.select(...)` | keeps only the listed columns | rows containing those columns | narrower schema |
| `.drop(...)` | removes the listed columns | any rows | narrower schema |
| `.rename(...)` | renames columns | rows containing the referenced columns | same values, renamed schema |
| `.cast(...)` | casts columns to new dtypes | rows containing the referenced columns | same rows, new column dtypes |

## Example

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

## Namespaces

### String

- `.lower()`
- `.upper()`
- `.strip()`
- `.len()`
- `.contains(...)`
- `.startswith(...)`
- `.endswith(...)`
- `.regex_contains(...)`
- `.replace(...)`
- `.regex_replace(...)`

### Datetime

- `.year()`
- `.month()`
- `.day()`
- `.hour()`
- `.to_date()`

## Mixed Pipelines

Expression-backed and Python-backed steps can be mixed:

```python
pipeline = (
    mdr.from_items([{"x": 1}, {"x": 2}])
    .with_columns(y=mdr.col("x") + 1)
    .map(lambda row: {"z": int(row["y"]) * 10})
    .with_column("w", mdr.col("z") + 5)
)
```

Refiner handles the boundary between Arrow blocks and row execution internally.

## Notes

- adjacent expression-backed operations are fused into vectorized segments
- vectorized execution is shard-local
- global operators like join, sort, and shuffle are still out of scope here

## Related Pages

- [Pipeline basics](pipeline-basics.md)
- [In-process debugging](in-process-debugging.md)
