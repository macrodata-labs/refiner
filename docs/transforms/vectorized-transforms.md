---
title: "Vectorized transforms"
description: "Use Arrow-backed columnar transforms in Refiner"
---

# Vectorized transforms

Vectorized transforms operate on columns instead of Python row objects. Refiner
fuses adjacent vectorized operations so data can stay in Arrow tables longer.

## Select, drop, rename

```python
pipeline = (
    pipeline
    .select("episode_id", "task", "frame_count")
    .rename(frame_count="num_frames")
    .drop("task")
)
```

## Computed columns

```python
pipeline = pipeline.with_columns(
    is_long=mdr.col("num_frames") > 300,
)
```

## Vectorized filter

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

## Custom Arrow table transforms

Use `map_table` when a library already accepts and returns Arrow tables:

```python
import pyarrow as pa


def normalize(table: pa.Table) -> pa.Table:
    return table


pipeline = pipeline.map_table(normalize)
```

For expensive state such as a GPU model, provide a zero-argument factory.
Refiner calls it once in each worker and reuses the returned callable for every
table handled by that worker:

```python
class Embedder:
    def __init__(self):
        self.model = load_model("clip").cuda()

    def __call__(self, table: pa.Table) -> pa.Table:
        embeddings = self.model.encode(table.column("image"))
        return table.append_column("embedding", embeddings)


def create_embedder():
    return Embedder()


pipeline = pipeline.map_table(factory=create_embedder)
```

The factory is not called while building, inspecting, or serializing the
pipeline. Use `with_max_block_rows(...)` and
`with_max_vectorized_block_bytes(...)` to bound the tables passed to the model.
Exactly one of the positional callback or `factory` must be provided.

## When not to use vectorized transforms

Use row transforms when you need:

- `row.videos`
- `row.to_frame_table()`
- Python libraries that operate on full episodes
- network/API calls
- mutation of nested frame data

Use vectorized transforms for simple column-level changes before or after row
operations.

## Related pages

- [Expressions](expressions.md)
- [Schemas and DTypes](schemas-and-dtypes.md)
- [Episode Rows](../episode-data/episode-rows.md)
