---
title: "Transforms"
description: "How Refiner transform methods work"
---

Transforms are the operations you run between reading data and writing it back out.

Refiner supports two execution styles:

- Python transforms, which run on rows or batches in Python
- expression-backed transforms, which run on Arrow data with the expression DSL

Use the expression-backed methods when you can. They are easier to optimize and can run on the vectorized path. Use Python transforms when you need arbitrary logic that does not fit the DSL.

## Transform methods

| method | accepted argument | how it runs | effect |
| --- | --- | --- | --- |
| `.map(fn)` | `fn(row) -> dict[str, object] \| Row` | Python row-by-row | patches or replaces each row |
| `.map_table(fn)` | `fn(table) -> pa.Table` | vectorized block-by-block | transforms fused Arrow tables directly |
| `.flat_map(fn)` | `fn(row) -> Iterable[dict[str, object] \| Row]` | Python row-by-row | emits zero, one, or many rows per input row |
| `.filter(predicate)` | `predicate` can be an `Expr` or `fn(row) -> bool` | expression-backed if given an `Expr`, otherwise Python row-by-row | removes rows that do not satisfy the predicate |
| `.batch_map(fn, batch_size=...)` | `fn(batch) -> Iterable[Row]` | Python batch-by-batch | transforms rows in batches instead of one at a time |
| `.map_async(fn, max_in_flight=..., preserve_order=...)` | async `fn(row) -> dict[str, object] \| Row` | Python async row-by-row | runs async enrichment or inference with bounded concurrency |
| `.with_columns(**assignments)` | keyword assignments where each value is an `Expr` or literal | expression-backed | adds or overwrites multiple columns |
| `.with_column(name, value)` | a column name plus an `Expr` or literal | expression-backed | adds or overwrites one column |
| `.select(*columns)` | one or more column names | expression-backed | keeps only the named columns |
| `.drop(*columns)` | one or more column names | expression-backed | removes named columns |
| `.rename(**mapping)` | keyword mapping from old name to new name | expression-backed | renames columns without changing values |
| `.cast(**dtypes)` | keyword mapping from column name to dtype string | expression-backed | changes column dtypes |

## What Your Function Receives

Python transforms receive `Row` objects, not raw dictionaries or Arrow tables.

- `map(...)` gets one `Row`
- `flat_map(...)` gets one `Row`
- Python `filter(...)` gets one `Row`
- `map_async(...)` gets one `Row`
- `batch_map(...)` gets `list[Row]`
- `map_table(...)` gets one `pa.Table`

Those rows are mapping-like and immutable from the caller's perspective:

- `row["col"]` works
- `dict(row.items())` works
- `row.update(...)` returns a patched row
- `row.drop(...)` returns a row with hidden keys

Do not assume:

- every row is a plain `dict`
- mutating a row in place is part of the API
- Python row UDFs receive internal `Tabular` blocks directly

`map_table(...)` is the explicit escape hatch when you do want block-level
access to the underlying Arrow table.

## How `map(...)` behaves

`map(...)` is patch-oriented by default.

If your function returns a mapping, Refiner merges it into the current row:

```python
pipeline = pipeline.map(lambda row: {"preview": row["text"][:80]})
```

If you prefer to be explicit, you can also mutate the row helper and return it:

```python
pipeline = pipeline.map(lambda row: row.update(preview=row["text"][:80]))
```

Use `map(...)` when you need arbitrary Python logic that does not fit the expression DSL.

## How `map_table(...)` behaves

`map_table(...)` is the block-level counterpart to `map(...)`.

It receives the fused underlying `pa.Table` on the vectorized path:

```python
import pyarrow as pa

pipeline = pipeline.map_table(
    lambda table: table.select(["id", "text"])
)
```

Use it when:

- you want direct access to the Arrow table
- the expression DSL is not enough
- you still want to stay on the fused vectorized execution path

The contract is:

- input: `pa.Table`
- output: `pa.Table`

Unlike `map(...)`, this is not a row UDF. It operates on internal vectorized
blocks, so block sizes are an execution detail and should not be treated as a
stable semantic unit.

Be careful with internal routing columns:

- do not drop or rewrite `__shard_id` unless you intentionally mean to change shard routing
- do not assume one `map_table(...)` call equals one shard; a block can contain rows from multiple shard ids

## How `filter(...)` behaves

`filter(...)` has two modes.

Expression-backed filter:

```python
pipeline = pipeline.filter(mdr.col("lang") == "en")
```

This stays on the vectorized Arrow path and is usually what you want for simple predicates.

That includes list-valued membership checks:

```python
pipeline = pipeline.filter(~mdr.col("tasks").is_in(["pick"]))
```

For list-valued columns like `tasks`, this means "keep rows whose list does not
contain any of these values."

Python filter:

```python
pipeline = pipeline.filter(lambda row: row["lang"] == "en")
```

This runs row-by-row in Python and is useful when the predicate needs custom logic.

## Vectorized transforms

These methods stay on the vectorized execution path:

- `.filter(expr)`
- `.map_table(...)`
- `.with_columns(...)`
- `.with_column(...)`
- `.select(...)`
- `.drop(...)`
- `.rename(...)`
- `.cast(...)`

Example:

```python
import refiner as mdr

pipeline = (
    mdr.read_jsonl("input/*.jsonl")
    .filter(mdr.col("lang") == "en")
    .with_columns(
        text=mdr.col("text").str.strip(),
        text_len=mdr.col("text").str.len(),
    )
    .select("text", "text_len")
)
```

See [Expressions](expressions.md) for the full DSL.

## Python transforms

Use Python transforms when you need custom logic or external libraries:

```python
def add_preview(row):
    return row.update(preview=" ".join(row["text"].split()[:20]))

pipeline = pipeline.map(add_preview)
```

Batch transforms are useful when one function call should handle many rows at once:

```python
pipeline = pipeline.batch_map(
    lambda batch: [row for row in batch if row["score"] > 0],
    batch_size=64,
)
```

Async transforms are useful for remote lookups or model calls:

```python
endpoint = mdr.inference.OpenAIEndpointProvider(
    base_url="https://api.openai.com",
    api_key="YOUR_API_KEY",
    model="gpt-5-mini"
)

pipeline = pipeline.map_async(
    mdr.inference.generate(
        fn=my_inference_fn,
        provider=endpoint,
    ),
    max_in_flight=64,
)
```

The same contract applies to richer row subclasses like `LeRobotRow`: the row
may expose extra helpers, but it still enters your Python function through the
normal row API.

## Related pages

- [Expressions](expressions.md)
- [Inference](inference.md)
- [Reading and writing data](reading-and-writing.md)
- [Pipeline basics](pipeline-basics.md)
- [Task pipelines](task-pipelines.md)
