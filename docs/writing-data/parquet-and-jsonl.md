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
    assets=mdr.FileAssetConfig(),
)
```

Use `BlobAssetConfig` to pack many assets into large range-addressable block
files instead of creating one output file per asset:

```python
pipeline.write_parquet(
    "/tmp/output",
    assets=mdr.BlobAssetConfig(target_bytes=1 << 30),
)
```

Here `1 << 30` means a 1 GiB target block size. Use `assets=None` (the
default) to leave the existing asset representation unchanged.

See [Media Assets and Reducers](media-assets-and-reducers.md).
