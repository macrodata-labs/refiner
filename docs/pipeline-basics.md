---
title: "Pipeline Basics"
description: "Overview of the basic Refiner pipeline model"
---

A Refiner pipeline has three parts:

- one source
- zero or more ordered transforms
- an optional sink

## Start With A Source

```python
import refiner as mdr

pipeline = mdr.read_parquet("data/*.parquet")
```

Other entry points:

### `from_items(...)`

```python
import refiner as mdr

pipeline = mdr.from_items(
    [
        {"text": "hello", "lang": "en"},
        {"text": "bonjour", "lang": "fr"},
    ]
)
```

If you pass mappings, those become rows directly. If you pass primitives like
strings or ints, Refiner wraps each value as `{"item": ...}`.

### `from_source(...)`

```python
pipeline = mdr.from_source(my_source)
```

### `task(...)`

```python
import refiner as mdr

pipeline = mdr.task(
    lambda rank, world_size: {
        "rank": rank,
        "world_size": world_size,
    },
    num_tasks=8,
)
```

Each task callback receives `rank` and `world_size`, so this is the right entry
point when you want one unit of work per task instead of iterating an existing
dataset.

See [Reading and writing data](reading-and-writing.md) for the different source
entry points and the row shapes they produce.

## Add Transforms

Transforms sit between the source and the sink.

Some transforms run row-by-row in Python:

- `.map(...)`
- `.flat_map(...)`
- `.batch_map(...)`
- `.map_async(...)`

Others use the expression DSL and can stay on the vectorized Arrow path:

- `.filter(expr)`
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
    pipeline
    .filter(mdr.col("lang") == "en")
    .with_columns(
        text=mdr.col("text").str.strip(),
        text_len=mdr.col("text").str.len(),
    )
)
```

See:

- [Transforms](transforms.md)
- [Expressions](expressions.md)

## What Your Python Code Gets

For Python UDF steps, Refiner passes `Row` objects into your function:

- `map(...)` gets one `Row`
- `map_async(...)` gets one `Row`
- `flat_map(...)` gets one `Row`
- Python `filter(...)` gets one `Row`
- `batch_map(...)` gets `list[Row]`

Those rows behave like immutable mappings:

```python
def normalize(row):
    return row.update(text=row["text"].strip())
```

Assumptions that are safe:

- `row["col"]` works
- `dict(row.items())` works
- `row.update(...)` / `row.drop(...)` return modified rows

Assumptions that are not safe:

- every row is a plain `dict`
- mutating a row in place is part of the API
- Python UDFs receive raw Arrow tables

If you want Arrow-backed/vectorized execution, use the expression API instead of
Python row functions.

## Add A Sink

Attach a sink when you want launched execution to write output:

- `.write_jsonl(...)`
- `.write_parquet(...)`
- `.write_lerobot(...)`

See [Reading and writing data](reading-and-writing.md) for the built-in readers
and writers.

## Example

```python
import refiner as mdr

def add_preview(row):
    return row.update(
        preview=" ".join(row["text"].split()[:20]),
    )

pipeline = (
    mdr.read_jsonl("input/*.jsonl")
    .filter(mdr.col("lang") == "en")
    .with_columns(
        text=mdr.col("text").str.strip(),
        text_len=mdr.col("text").str.len(),
    )
    .map(add_preview)
    .write_parquet("s3://my-bucket/english-cleanup/")
)
```

## Step Contracts

### `map(...)`

May return:

- a `Row`
- a `Mapping[str, Any]`

For `map(...)`, returning a `Mapping[str, Any]` is treated as a patch and merged
into the input row.

### `flat_map(...)` and `batch_map(...)`

`flat_map(...)` should emit zero or more row-like items:

- `Row`
- `Mapping[str, Any]`
- `None` to drop an item

`batch_map(...)` receives `list[Row]` and should emit the same kinds of items.
It does not receive internal `Tabular` blocks directly.

## Related Pages

- [Transforms](transforms.md)
- [Expressions](expressions.md)
- [Reading and writing data](reading-and-writing.md)
- [In-process debugging](in-process-debugging.md)
- [Launchers](launchers.md)
- [Task pipelines](task-pipelines.md)
