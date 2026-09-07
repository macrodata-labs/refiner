---
title: "Tabular files"
description: "Read Parquet, JSON, JSONL, and CSV files"
---

# Tabular files

Use tabular readers when your input already has row-shaped records.

## Parquet

```python
pipeline = mdr.read_parquet(
    "/data/episodes/*.parquet",
    file_path_column="source_file",
)
```

Parquet is the best general-purpose tabular format when you control the data.
It preserves schema information and works well with vectorized transforms.

For rows with large or variable-sized values, set the pipeline-wide execution
block limit:

```python
pipeline = mdr.read_parquet("/data/episodes/*.parquet").with_max_block_rows(256)
```

Refiner passes this limit to the Parquet scanner and also enforces it on
downstream execution blocks. Parquet may still emit smaller batches at file or
row-group boundaries.

To tune Parquet scanning independently, set `read_batch_rows`:

```python
pipeline = mdr.read_parquet(
    "/data/episodes/*.parquet",
    read_batch_rows=4_096,
).with_max_block_rows(256)
```

The explicit read setting controls Parquet scanner batches; the pipeline limit
continues to cap downstream execution blocks.

## JSON and JSONL

```python
json_pipeline = mdr.read_json("/data/episodes/*.json")
jsonl_pipeline = mdr.read_jsonl("/data/episodes/*.jsonl")
```

Use JSONL for large line-delimited datasets. It can be split more naturally
than one large JSON document.

## CSV

```python
pipeline = mdr.read_csv(
    "/data/annotations/*.csv",
    multiline_rows=False,
)
```

CSV is useful for metadata and labels. Prefer Parquet for nested arrays, frame
data, or typed feature columns.

## DTypes

Use `dtypes` when the source format needs help describing a column:

```python
pipeline = mdr.read_parquet(
    "/data/videos.parquet",
    dtypes={"video": mdr.datatype.video_path()},
)
```

See [Schemas and DTypes](../transforms/schemas-and-dtypes.md).

## Related pages

- [Vectorized Transforms](../transforms/vectorized-transforms.md)
- [Parquet and JSONL Writers](../writing-data/parquet-and-jsonl.md)
