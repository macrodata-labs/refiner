---
title: "Pipeline Basics"
description: "Build Refiner pipelines from sources, transforms, and sinks"
---

A Refiner pipeline is:

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

Use this for small in-memory inputs:

```python
import refiner as mdr

pipeline = mdr.from_items(
    [
        {"text": "hello", "lang": "en"},
        {"text": "bonjour", "lang": "fr"},
    ]
)
```

### `from_source(...)`

Use this when you already have a custom source object:

```python
pipeline = mdr.from_source(my_source)
```

### `task(...)`

Use this when you want to run arbitrary task-style work instead of reading a dataset.

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

## Add Transforms

### Python row transforms

```python
pipeline = pipeline.map(lambda row: {"x": row["x"] + 1})
pipeline = pipeline.filter(lambda row: row["x"] > 0)
pipeline = pipeline.flat_map(lambda row: [row, {"x": row["x"] * 10}])
```

For `map(...)`, returning a `Mapping[str, Any]` is treated as a patch and merged
into the input row.

```python
pipeline = pipeline.map(lambda row: {"length_bucket": "long"})
```

If you want to be explicit, return `row.update(...)` instead:

```python
pipeline = pipeline.map(lambda row: row.update(length_bucket="long"))
```

### Batch transforms

```python
pipeline = pipeline.batch_map(
    lambda batch: [row for row in batch if row["x"] > 0],
    batch_size=64,
)
```

### Async row transforms

```python
pipeline = pipeline.map_async(fetch_enrichment, max_in_flight=32)
```

### Vectorized expression transforms

Use expression-backed transforms when you want Arrow-backed execution:

```python
import refiner as mdr

pipeline = (
    pipeline
    .filter(mdr.col("lang") == "en")
    .with_columns(
        text=mdr.col("text").str.strip(),
        text_len=mdr.col("text").str.len(),
    )
    .select("text", "text_len")
)
```

Common expression-backed methods:

- `.filter(expr)`
- `.with_columns(...)`
- `.with_column(...)`
- `.select(...)`
- `.drop(...)`
- `.rename(...)`
- `.cast(...)`

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

Built-in sinks:

- `.write_jsonl(...)`
- `.write_parquet(...)`
- `.write_lerobot(...)`

Example:

```python
pipeline = pipeline.write_parquet("out/")
```

Attaching a sink affects launched execution. In-process debugging helpers still
return rows and do not write output.

## Example

```python
import refiner as mdr

pipeline = (
    mdr.read_jsonl("input/*.jsonl")
    .filter(mdr.col("lang") == "en")
    .with_columns(
        text=mdr.col("text").str.strip(),
        text_len=mdr.col("text").str.len(),
    )
    .map(lambda row: row.update(length_bucket="long" if row["text_len"] > 512 else "short"))
    .write_parquet("s3://my-bucket/english-cleanup/")
)
```

## Step Contracts

### `map(...)`

May return:

- a `Row`
- a `Mapping[str, Any]`

Use `filter(...)` to drop rows and `flat_map(...)` to emit `0..N` rows.

### `flat_map(...)` and `batch_map(...)`

`flat_map(...)` should emit zero or more row-like items:

- `Row`
- `Mapping[str, Any]`
- `None` to drop an item

`batch_map(...)` receives `list[Row]` and should emit the same kinds of items.
It does not receive internal `Tabular` blocks directly.

## Related Pages

- [Expression transforms](expression-transforms.md)
- [Readers and sharding](readers-and-sharding.md)
- [In-process debugging](in-process-debugging.md)
- [Launchers](launchers.md)
- [Task pipelines](task-pipelines.md)
