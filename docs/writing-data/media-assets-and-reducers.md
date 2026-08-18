---
title: "Media Assets and Reducers"
description: "How Refiner writers handle media files, assets, and reducer stages"
---

# Media Assets And Reducers

Robotics datasets often contain large media values. Refiner distinguishes the
row value from the asset storage behavior through dtypes and video source APIs.

## Asset Columns

Use dtypes to mark media columns:

```python
pipeline = mdr.read_parquet(
    "/data/videos.parquet",
    dtypes={"video": mdr.datatype.video_path()},
)
```

Writers can then copy or upload assets instead of treating them as plain strings.

## Missing Asset Policy

Parquet and JSONL writers support `missing_asset_policy`:

```python
pipeline.write_parquet(
    "/tmp/out",
    upload_assets=True,
    missing_asset_policy="error",
)
```

Use `"error"` for training data. Missing media should usually fail the job.

## Reducers

Some writers add a reducer stage. A reducer stage finalizes outputs after
workers finish shard-local writes.

| Writer | Reducer purpose |
| --- | --- |
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
