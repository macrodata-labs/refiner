---
title: "Expressions"
description: "Refiner's expression DSL for vectorized transforms"
---

# Expressions

Expressions describe columnar computations for vectorized transforms.

## Constructors

```python
mdr.col("episode_id")
mdr.lit("train")
```

## Boolean Logic

Use `&`, `|`, and `~`. Do not use Python `and` or `or`.

```python
pipeline = pipeline.filter(
    (mdr.col("num_frames") > 30) & (mdr.col("split") == "train")
)
```

## Conditional Values

```python
pipeline = pipeline.with_columns(
    bucket=mdr.if_else(mdr.col("num_frames") > 300, "long", "short")
)
```

## Null Handling

```python
pipeline = pipeline.with_columns(
    task=mdr.coalesce(mdr.col("task"), mdr.lit("unknown"))
)
```

## String And Datetime Helpers

Expression objects include namespaces for common string and datetime operations
when supported by the underlying Arrow execution.

```python
pipeline = pipeline.filter(mdr.col("episode_id").str.contains("pick"))
```

## Related Pages

- [Vectorized Transforms](vectorized-transforms.md)
- [Schemas and DTypes](schemas-and-dtypes.md)

