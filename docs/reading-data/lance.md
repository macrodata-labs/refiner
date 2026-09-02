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
    num_shards=32,
    max_rows=2_000,
)
```

When `version` is omitted, Refiner resolves the latest version once and pins it
for the pipeline. Column projection is pushed into Lance.

Use the pipeline-wide execution block limit when rows contain large media or
other variable-sized values:

```python
pipeline = mdr.load_lance("s3://my-bucket/hands.lance").with_max_block_rows(128)
```

Refiner propagates this limit into Lance's scanner, so the source does not
materialize a larger Arrow batch and then immediately split it. The same limit
continues to apply to blocks produced by downstream transforms. Omit it to use
Lance's internal read-batch default.

When source rows are lightweight references but downstream transforms expand
them into large media values, tune the two boundaries independently:

```python
pipeline = mdr.load_lance(
    "s3://my-bucket/hands.lance",
    read_batch_rows=256,
).with_max_block_rows(8)
```

Here Lance scans 256 rows at a time while transforms and sinks receive blocks
of at most eight rows. An explicit `read_batch_rows` value takes precedence over
scanner inheritance from `with_max_block_rows`.

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
For large assets such as video, stream the byte range into a consumer instead:

```python
import av

with mdr.open_blob_stream(row["video"]) as stream:
    with av.open(stream, mode="r") as container:
        frames = list(container.decode(video=0))
```

`open_blob_stream(...)` is non-seekable and context-managed. It reads 8 MiB
chunks by default and buffers at most four queued chunks, applying backpressure
when the consumer is slower than storage. Set `chunk_bytes` and
`prefetch_chunks` when a workload needs different bounds. Closing the stream
early cancels pending prefetch work and releases its producer thread.
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
