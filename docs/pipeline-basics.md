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

You can also start from a custom source:

```python
pipeline = mdr.from_source(my_source)
```

## Add Transforms

Row transform:

```python
pipeline = pipeline.map(lambda row: {"x": row["x"] + 1})
```

Filter:

```python
pipeline = pipeline.filter(lambda row: row["x"] > 0)
```

Batch transform:

```python
pipeline = pipeline.batch_map(
    lambda batch: [row for row in batch if row["x"] > 0],
    batch_size=64,
)
```

Expansion:

```python
pipeline = pipeline.flat_map(lambda row: [row, {"x": row["x"] * 10}])
```

Async row transform:

```python
pipeline = pipeline.map_async(fetch_enrichment, max_in_flight=32)
```

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

Attaching a sink affects launched execution. Local iteration helpers like `iter_rows()` and `materialize()` still return rows and do not write output.

## Example

```python
import refiner as mdr

pipeline = (
    mdr.read_jsonl("input/*.jsonl")
    .filter(lambda row: row["lang"] == "en")
    .map(lambda row: {"text": row["text"].strip()})
    .write_parquet("out/")
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

### Expression-backed transforms

Use the expression API when you want Arrow-backed vectorized execution:

- `.select(...)`
- `.with_columns(...)`
- `.with_column(...)`
- `.drop(...)`
- `.rename(...)`
- `.cast(...)`
- `.filter(expr)`

See [Expression transforms](expression-transforms.md).

## Related Pages

- [Readers and sharding](readers-and-sharding.md)
- [Expression transforms](expression-transforms.md)
- [Local execution](local-execution.md)
- [Launchers](launchers.md)
