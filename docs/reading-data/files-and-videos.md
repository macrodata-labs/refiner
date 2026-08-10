---
title: "Files and videos"
description: "Read file inventories, raw bytes, and video paths"
---

# Files and videos

Use `read_files` to list files or read bytes. Use `read_videos` when the files
are video sources that should be typed as video assets.

## File inventory

```python
pipeline = mdr.read_files(
    "/data/raw/**/*",
    recursive=True,
    file_path_column="path",
    size_column="size",
)
```

This emits one row per file without reading file contents.

## File bytes

```python
pipeline = mdr.read_files(
    "/data/prompts/*.txt",
    content_column="text",
    decode_fn=lambda data: data.decode("utf-8"),
)
```

Use `content_column` only when the contents are small enough to process as row
values. For large media, keep paths and let media-aware APIs stream the data.

## Video files

```python
pipeline = (
    mdr.read_videos("/data/videos/*.mp4", file_path_column="video")
    .to_robot_rows(video_keys=("video",), fps=30.0)
)
```

`read_videos` creates a video-typed column, which lets `to_robot_rows` expose it
through `row.videos`.

## Related pages

- [Video Sources](../episode-data/frames-and-videos.md)
- [Media Assets and Reducers](../writing-data/media-assets-and-reducers.md)

