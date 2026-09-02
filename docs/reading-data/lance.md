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
    max_rows=2_000,
)
```

When `version` is omitted, Refiner resolves the latest version once and pins it
for the pipeline. Column projection is pushed into Lance, and `batch_size`
controls the streamed Arrow batch size.

## Limit the number of rows

Set `max_rows` to a non-negative integer to read only that many leading rows from
the pinned dataset version. Refiner uses the same global bounded-source wrapper
as other built-in readers, stops after the final required Arrow batch, and
slices that batch when necessary. Datasets with fewer rows simply yield all
available rows.

The limit is applied before pipeline transforms. Omit `max_rows` to read the
entire pinned version, or use `max_rows=0` to produce no source rows. For
example, this processes only the first 10,000 stored rows:

```python
pipeline = mdr.load_lance(
    "s3://my-bucket/hands.lance",
    max_rows=10_000,
)
```

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

By default, Refiner creates one shard per Lance fragment up to the 1,000-shard
automatic limit. For datasets with more fragments, it groups adjacent fragments
without dropping any data. Set `num_shards` to request up to 10,000 scheduling
units explicitly. Fragments remain atomic, so requesting more shards than
fragments still produces one shard per fragment. A worker may claim and process
multiple shards over its lifetime.
