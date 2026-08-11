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
    num_shards=32,
)
```

When `version` is omitted, Refiner resolves the latest version once and pins it
for the pipeline. Column projection is pushed into Lance, and `batch_size`
controls the streamed Arrow batch size.

Classic Lance blob columns are returned as lazy Refiner blob references:

```python
row = pipeline.take(1)[0]
encoded = mdr.read_blob(row["image"])
```

Each reference contains `path`, `offset`, and `size`. `read_blob(...)` performs
an exact byte-range read, so scanning rows does not materialize the blob bytes.
The reference remains valid only while the pinned dataset version's data files
are retained.
Lance Blob V2 columns are currently rejected because packed and dedicated V2
storage does not expose a stable physical object path that Refiner can preserve
in this three-field representation.

`load_lance` uses Lance's native storage layer. It rejects configured fsspec
filesystem objects and `storage_options`; provide a URI whose endpoint and
credentials are available through provider-standard environment configuration
to both Lance and fsspec. Credential-bearing URIs and URI query parameters are
rejected so pipeline plans do not serialize secrets.

By default, each Lance fragment becomes one Refiner shard. Set `num_shards` to
group adjacent fragments into fewer scheduling units. Fragments remain atomic,
so requesting more shards than fragments still produces one shard per fragment.
A worker may claim and process multiple shards over its lifetime.
