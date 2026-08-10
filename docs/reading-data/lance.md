---
title: "Lance"
description: "Read immutable Lance dataset versions as fragment-aligned Refiner shards"
---

# Lance

Lance support is optional:

```bash
pip install macrodata-refiner[lance]
```

Use `load_lance(...)` to read a pinned Lance dataset version:

```python
import refiner as mdr

pipeline = mdr.load_lance(
    "s3://my-bucket/hands.lance",
    version=42,
    columns=["image", "frame_id"],
    batch_size=128,
)
```

When `version` is omitted, Refiner resolves the latest version once and pins it
for the pipeline. Column projection is pushed into Lance, and `batch_size`
controls the streamed Arrow batch size. Use `blob_handling` to select Lance's
blob materialization behavior when reading blob columns.

`load_lance` uses Lance's native storage layer. It rejects configured fsspec
filesystem objects and `storage_options`; provide a URI whose endpoint and
credentials are available to Lance instead.

Each Lance fragment becomes one Refiner shard. A worker may claim and process
multiple fragments over its lifetime.

## Internal Notes

The source keeps the dataset URI and resolved version on the pipeline. It uses
ordinary row-range shards to assign fragment indices and attaches protected
fragment-ID and fragment-local-row-position columns to the rows. The
`add_columns` writer uses those columns to restore output order and validate
one-to-one row alignment.
