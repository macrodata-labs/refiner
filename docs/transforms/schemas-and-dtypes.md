---
title: "Schemas and DTypes"
description: "Use dtype metadata for assets, media, and writer schemas"
---

# Schemas and DTypes

DTypes attach semantic metadata to columns. This matters most for assets and
media, where a string or bytes column needs to be treated as a file, image,
audio, video, or PDF.

## Video columns

```python
pipeline = mdr.read_parquet(
    "/data/video_paths.parquet",
    dtypes={"video": mdr.datatype.video_path()},
)
```

Now downstream transforms can expose `video` through `to_robot_rows`:

```python
pipeline = pipeline.to_robot_rows(
    episode_id_key="episode_id",
    video_keys=("video",),
    fps=30.0,
)
```

## Asset types

| Helper | Use it for |
| --- | --- |
| `file_path()` / `file_bytes()` | Generic files. |
| `image_path()` / `image_bytes()` | Images. |
| `audio_path()` / `audio_bytes()` | Audio. |
| `video_path()` / `video_bytes()` | Encoded videos. |
| `video_frame_array()` | In-memory RGB frame arrays. |
| `pdf_path()` / `pdf_bytes()` | PDFs. |

## Transform output DTypes

```python
def add_clip(row):
    return row.update({"clip": row.videos["front"].clipped(to_timestamp_s=2.0)})


pipeline = pipeline.map(
    add_clip,
    dtypes={"clip": mdr.datatype.video_path()},
)
```

## Related pages

- [Files and Videos](../reading-data/files-and-videos.md)
- [Media Assets and Reducers](../writing-data/media-assets-and-reducers.md)

