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

The same configurations work with JSONL and Lance writers. These are generic
Refiner block files, not Lance-native blob columns. With `assets=None`, writers
preserve each asset's existing path, bytes, or reference representation.

The configuration fields are:

| Configuration | Fields |
| --- | --- |
| `FileAssetConfig` | `subdir`, `max_in_flight`, `missing_policy` |
| `BlobAssetConfig` | `subdir`, `target_bytes`, `missing_policy` |

`FileAssetConfig.max_in_flight` bounds concurrent asset uploads per writer. The
upload window stays active across input blocks, so even one-row blocks can
overlap when `max_in_flight` is greater than one. Refiner applies backpressure
when the window is full and waits for all remaining uploads when the shard is
finalized. Output rows are kept in order and are not written until every asset
referenced by that row's input block has finished uploading.

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

## MP4 Video Output on S3

When a video-writing pipeline targets an `s3://` output, Refiner writes a
standard indexed MP4 directly to the destination object. The completed file
contains a normal `moov` index (rather than fragmented `moof` segments), so
video readers can determine the full duration and frame count and seek across
the complete video.

No staging directory is required. Refiner holds the first 5 MiB of the output
and one active 5 MiB upload part in memory while it streams later
multipart-upload parts. This allows it to write the final MP4 media-data size
and index when the video closes without buffering the whole video locally.
Small videos, up to 5 MiB, are uploaded as a single object.

This works with AWS S3 and Cloudflare R2 when R2 is configured through the
`s3fs` S3-compatible endpoint. Once the first 5 MiB is available, Refiner
uploads multipart part 1. At close, it uploads the corrected prefix again as
part 1 before completing the multipart upload. R2 requires all non-final
parts to have the same size; Refiner uses fixed 5 MiB parts for that reason.

The S3 principal needs permission to put objects and, for videos larger than
5 MiB, to create, upload, complete, and abort multipart uploads. If the final
video close fails, Refiner aborts the multipart upload; the destination object
is not made visible by that upload.
