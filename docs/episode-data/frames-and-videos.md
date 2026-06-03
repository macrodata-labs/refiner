---
title: "Frames and Videos"
description: "Work with frame tables, video sources, and clips"
---

# Frames And Videos

An episode can contain both frame tables and videos:

- frame tables hold aligned values such as timestamp, action, state, and labels
- videos are lazy media sources that can be decoded, clipped, remuxed, or transcoded

## Frame Tables

```python
row = pipeline.take(1)[0]
frames = row.to_frame_table()

print(frames.names)
print(frames.num_rows)
```

Frame tables are Arrow-backed, which makes them efficient for slicing and
writing. Common frame columns include:

| Column | Meaning |
| --- | --- |
| `timestamp` | Seconds from episode start. |
| `action` | Robot action vector. |
| `observation.state` | Robot state vector. |
| `observation.*` | Additional observations. |
| `task_index` | LeRobot task index for each frame. |

## Observations

```python
state = row.observations("state")
all_observations = row.observations()
```

`observations()` normalizes names: `row.observations("state")` reads
`observation.state`.

## Video Sources

```python
for key, video in row.videos.items():
    print(key, video)
```

Video sources may be path-backed files, bytes, in-memory frame arrays, or frame
sequences. They share the same core operations:

| Operation | Purpose |
| --- | --- |
| `get_frame_count()` | Return the exact frame count when it is known or reported by the encoded container. |
| `clipped(...)` | Create a time-bounded view. |
| `iter_frames()` | Decode video frames lazily. |
| `iter_numpy_frames()` | Decode RGB arrays lazily. |
| `write_to(...)` | Let a writer copy/remux/transcode media. |

## Video Frame Counts

```python
async def add_frame_count(row):
    video = row.videos["observation.images.top"]
    return row.update(video_frame_count=await video.get_frame_count())
```

`get_frame_count()` is explicit because path-backed and bytes-backed videos may
need to open the encoded container. In-memory frame arrays return their exact
length. Frame sequences return their provided count, or count the repeatable
sequence when no count was provided. Encoded videos use container metadata and
raise an error when the container does not report an exact frame count.

## Clip A Video View

```python
def keep_middle(row):
    video = row.videos["observation.images.top"]
    return row.with_video(
        "observation.images.top",
        video.clipped(from_timestamp_s=1.0, to_timestamp_s=4.0),
    )
```

This does not immediately rewrite the media file. The writer later decides
whether to remux or transcode the selected time range.

## Related Pages

- [Files and Videos Reader](../reading-data/files-and-videos.md)
- [Media Assets and Reducers](../writing-data/media-assets-and-reducers.md)
- [Motion Trimming](../episode-operations/motion-trimming.md)
