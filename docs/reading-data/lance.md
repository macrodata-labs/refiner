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
controls the streamed Arrow batch size. Use `blob_handling` to select Lance's
blob materialization behavior when reading blob columns.

`load_lance` uses Lance's native storage layer. It rejects configured fsspec
filesystem objects and `storage_options`; provide a URI whose endpoint and
credentials are available through provider-standard environment configuration
to both Lance and fsspec. Credential-bearing URIs and URI query parameters are
rejected so pipeline plans do not serialize secrets.

By default, each Lance fragment becomes one Refiner shard. Set `num_shards` to
group adjacent fragments into fewer scheduling units. Fragments remain atomic,
so requesting more shards than fragments still produces one shard per fragment.
A worker may claim and process multiple shards over its lifetime.
