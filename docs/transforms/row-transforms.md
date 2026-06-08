---
title: "Row transforms"
description: "Use map, flat_map, and filters over episode rows"
---

# Row transforms

Use row transforms when each episode can be handled independently in Python.

## `map`

`map` takes one row and returns one row.

```python
def add_frame_count(row):
    return row.update({"frame_count": row.num_frames})


pipeline = pipeline.map(add_frame_count)
```

This is the default transform for simple episode enrichment.

## `filter`

Pass a Python predicate to keep or drop rows:

```python
pipeline = pipeline.filter(lambda row: row.num_frames > 30)
```

For simple column predicates, vectorized filters are often faster; see
[Vectorized Transforms](vectorized-transforms.md).

## `flat_map`

`flat_map` takes one row and returns zero or more rows.

```python
def split_by_camera(row):
    for camera, video in row.videos.items():
        yield row.update({"camera": camera, "video": video})


pipeline = pipeline.flat_map(split_by_camera)
```

Use this when one episode naturally becomes several examples.

## DType hints

When a transform creates a column that a writer should treat as an asset or
media value, pass `dtypes`:

```python
pipeline = pipeline.map(
    lambda row: row.update({"clip": row.videos["front"]}),
    dtypes={"clip": mdr.datatype.video_path()},
)
```

See [Schemas and DTypes](schemas-and-dtypes.md).

## Related pages

- [Episode Rows](../episode-data/episode-rows.md)
- [Motion Trimming](../episode-operations/motion-trimming.md)

