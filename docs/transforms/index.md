---
title: "Transforms"
description: "How to change, enrich, filter, and type episode rows in Refiner"
---

# Transforms

Transforms are pipeline steps between readers and writers. They can operate on
Python rows, async I/O, batches, or Arrow-backed vectorized tables.

| Transform style | Use it for |
| --- | --- |
| [Row Transforms](row-transforms.md) | Simple episode-level logic. |
| [Async and Batch Transforms](async-and-batch-transforms.md) | API calls, model calls, batched perception. |
| [Vectorized Transforms](vectorized-transforms.md) | Fast columnar filtering, projection, renaming, and computed columns. |
| [Expressions](expressions.md) | The expression DSL behind vectorized transforms. |
| [Schemas and DTypes](schemas-and-dtypes.md) | Type hints for assets, media, and writer schemas. |

For packaged episode-level workflows, see
[Episode Operations](../episode-operations/index.md).

