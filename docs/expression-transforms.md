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

The input to every expression-backed operation is an Arrow-backed table built from the current rows in the pipeline. In practice that means:

- every referenced column must already exist
- each expression is evaluated column-wise, not row-by-row in Python
- the output row shape is determined entirely by the expression method you call

| method | what it does | input shape | output shape |
| --- | --- | --- | --- |
| `.filter(expr)` | keeps only rows where `expr` evaluates truthy | rows containing the columns used by `expr` | same columns, fewer rows |
| `.with_columns(...)` | adds or overwrites multiple columns from expressions | rows containing the columns used by each expression | same rows, wider or updated schema |
| `.with_column(...)` | adds or overwrites one column from an expression | rows containing the columns used by the expression | same rows, one column added or replaced |
| `.select(...)` | keeps only the listed columns, in the requested order | rows containing those columns | narrower schema with reordered columns if requested |
| `.drop(...)` | removes the listed columns | any rows; dropped columns may or may not exist depending on usage | narrower schema |
| `.rename(...)` | renames existing columns | rows containing the referenced source columns | same values under new column names |
| `.cast(...)` | casts existing columns to new dtypes | rows containing the referenced columns | same rows, same columns, different dtypes |

Typical input rows look like ordinary row dictionaries:

```python
{"doc_id": "a1", "text": "hello world", "lang": "en"}
```

Expression-backed methods operate on that schema in bulk. For example:

- `.filter(mdr.col("lang") == "en")` expects a `lang` column
- `.with_column("text_len", mdr.col("text").str.len())` expects a `text` column
- `.cast(score="float64")` expects an existing `score` column

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
