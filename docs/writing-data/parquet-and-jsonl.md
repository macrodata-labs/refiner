---
title: "Parquet and JSONL writers"
description: "Write tabular and line-delimited Refiner outputs"
---

# Parquet and JSONL writers

Use Parquet for typed tabular output and JSONL for simple line-delimited rows.

## Parquet

```python
pipeline.write_parquet(
    "/tmp/output-parquet",
    compression="zstd",
)
```

Parquet is the better default for data you will read again as a dataset.

## JSONL

```python
pipeline.write_jsonl("/tmp/output-jsonl")
```

JSONL is useful for logs, model responses, or lightweight inspection output.

## Asset columns

Both writers can upload asset columns:

```python
pipeline.write_parquet(
    "/tmp/output",
    upload_assets=True,
    assets_subdir="assets",
)
```

See [Media Assets and Reducers](media-assets-and-reducers.md).

