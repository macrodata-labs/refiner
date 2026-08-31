---
title: "Media assets and reducers"
description: "How Refiner writers handle media files, assets, and reducer stages"
---

# Media assets and reducers

Robotics datasets often contain large media values. Refiner distinguishes the
row value from the asset storage behavior through dtypes and video source APIs.

## Asset columns

Use dtypes to mark media columns:

```python
pipeline = mdr.read_parquet(
    "/data/videos.parquet",
    dtypes={"video": mdr.datatype.video_path()},
)
```

Writers can then copy or upload assets instead of treating them as plain strings.

## Blob references

Range-addressable media can use a portable blob reference:

```python
field = mdr.datatype.blob_reference("image")
encoded = mdr.read_blob(row["image"])
```

The Arrow value is a `{path, offset, size}` struct. `read_blob(...)` reads only
that exact range.

## Asset output layouts

Use `FileAssetConfig` to materialize every recognized asset as an individual
file and store its path:

```python
pipeline.write_parquet(
    "/tmp/out",
    assets=mdr.FileAssetConfig(),
)
```

Use `BlobAssetConfig` to concatenate assets into worker-owned block files and
store `{path, offset, size}` references:

```python
pipeline.write_parquet(
    "/tmp/out",
    assets=mdr.BlobAssetConfig(target_bytes=1 << 30),
)
```

`1 << 30` is 1 GiB (1,073,741,824 bytes). `target_bytes` is a rollover
target, not a hard limit: an individual asset larger than the target is written
to its own larger block.

When the input rows collectively reference every byte of an existing packed
blob, Refiner copies that source blob once and translates the output offsets.
It does not download and upload each referenced range separately. Partial
selections retain the normal range-copy behavior, and the optimization is
disabled for missing-asset policies other than `"error"` so their row-level
semantics remain unchanged.

When the destination uses s3fs, including AWS S3 and Cloudflare R2, Refiner
uploads large files with bounded multipart concurrency. No additional
configuration is required. Small files retain s3fs's one-shot PUT behavior.
Credentials must permit object writes and multipart creation, part upload,
completion, and abort.

The same configurations work with JSONL and Lance writers. These are generic
Refiner block files, not Lance-native blob columns. With `assets=None`, writers
preserve each asset's existing path, bytes, or reference representation.

The configuration fields are:

| Configuration | Fields |
| --- | --- |
| `FileAssetConfig` | `subdir`, `max_in_flight`, `missing_policy` |
| `BlobAssetConfig` | `subdir`, `target_bytes`, `missing_policy` |

To migrate the previous individual-file API:

```python
# Before
pipeline.write_parquet(
    "/tmp/out",
    upload_assets=True,
    assets_subdir="media",
    max_asset_uploads_in_flight=32,
    missing_asset_policy="error",
)

# Now
pipeline.write_parquet(
    "/tmp/out",
    assets=mdr.FileAssetConfig(
        subdir="media",
        max_in_flight=32,
        missing_policy="error",
    ),
)
```

## Missing asset policy

Both configurations support a missing-asset policy:

```python
pipeline.write_parquet(
    "/tmp/out",
    assets=mdr.FileAssetConfig(missing_policy="error"),
)
```

Use `"error"` for training data. Missing media should usually fail the job.
`"set_null"` retains the row and replaces a missing asset with null.
`"drop_row"` removes the complete row containing a missing asset.

## Reducers

Some writers add a reducer stage. A reducer stage finalizes outputs after
workers finish shard-local writes.

| Writer | Reducer purpose |
| --- | --- |
| Parquet / JSONL / standalone Lance | Remove files and asset blocks from rejected worker attempts. |
| Lance dataset | Commit finalized fragments and remove rejected metadata and assets. |
| LeRobot | Merge metadata, tasks, stats, and staged chunks. |
| Zarr | Merge shard-local stores into a single store when configured. |

Reducers are part of the launched pipeline plan and are visible in job progress.

## Internal Notes

When a Refiner `DataFile` or `DataFolder` resolves an s3fs filesystem, Refiner
installs an idempotent, process-wide replacement for s3fs's `S3File` class.
Consequently, subsequent binary s3fs streaming writes in that worker use
concurrent multipart batches, not only packed asset writes. Reads and non-s3fs
fsspec backends retain their standard behavior.

Each writer accumulates up to four parts using the block size configured on
s3fs (50 MiB by default) and uploads the batch concurrently. Packed asset
writers pass their maximum final block size to s3fs, allowing the part size to
grow automatically so objects through S3's 5 TiB limit remain below 10,000
parts. A direct s3fs write with no final `size` hint retains its configured part
size and therefore cannot exceed 10,000 parts; it fails and aborts cleanly
instead. Peak memory includes the batch buffer and materialized input bytes.
The patch subclasses s3fs's file class and keeps its normal multipart creation,
commit, abort, endpoint, credential, and cache behavior.

Complete-blob coalescing groups scalar blob references by source path and only
uses the fast path when their sorted union covers the source from byte zero
through its exact size without gaps. Overlapping references are supported;
malformed or partial reference sets fall back to the established per-range
writer.
