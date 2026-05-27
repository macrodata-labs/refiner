---
title: "Vectorized Transforms"
description: "Use Arrow-backed columnar transforms in Refiner"
---

# Vectorized Transforms

Vectorized transforms operate on columns instead of Python row objects. Refiner
fuses adjacent vectorized operations so data can stay in Arrow tables longer.

## Select, Drop, Rename

```python
pipeline = (
    pipeline
    .select("episode_id", "task", "frame_count")
    .rename(frame_count="num_frames")
    .drop("task")
)
```

## Computed Columns

```python
pipeline = pipeline.with_columns(
    is_long=mdr.col("num_frames") > 300,
)
```

## Vectorized Filter

```python
pipeline = pipeline.filter(mdr.col("num_frames") > 30)
```

Pass an expression for vectorized filtering. Pass a Python callable for row-level
filtering.

## Casting

```python
pipeline = pipeline.cast(
    video=mdr.datatype.video_path(),
)
```

Use casts when the storage format cannot fully represent asset/media semantics.

## When Not To Use Vectorized Transforms

Use row transforms when you need:

- `row.videos`
- `row.to_frame_table()`
- Python libraries that operate on full episodes
- network/API calls
- mutation of nested frame data

Use vectorized transforms for simple column-level changes before or after row
operations.

## Related Pages

- [Expressions](expressions.md)
- [Schemas and DTypes](schemas-and-dtypes.md)
- [Episode Rows](../episode-data/episode-rows.md)

