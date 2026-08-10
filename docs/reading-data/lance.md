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

Each Lance fragment becomes one Refiner shard. A worker may claim and process
multiple fragments over its lifetime.

## Internal Notes

The source records the dataset URI, resolved version, fragment ID, and visible
row count in each shard descriptor. It also attaches a protected
fragment-local row position used by the `add_columns` writer to restore output
order and validate one-to-one row alignment.
