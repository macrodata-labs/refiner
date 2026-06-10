---
title: "Reader model"
description: "How Refiner readers plan shards and emit rows"
---

# Reader model

A Refiner reader has two jobs:

1. **Plan work** into shards.
2. **Read each shard** into rows or tabular blocks.

That separation is important. Planning should be lightweight enough to run at
submission time. Reading can do the expensive I/O inside workers.

## Rows and blocks

Most user code sees rows:

```python
for row in pipeline.take(3):
    print(row)
```

Internally, readers may emit either:

| Unit | Why it exists |
| --- | --- |
| Row | Natural for Python transforms and episode-level robotics data. |
| Tabular block | Efficient for Arrow-backed readers and vectorized operations. |

You usually do not need to choose. Refiner converts blocks to rows when a row
transform needs them and keeps Arrow tables when vectorized transforms can run
directly.

## Reader arguments you see often

| Argument | Meaning |
| --- | --- |
| `target_shard_bytes` | Approximate shard size for byte-splittable inputs. |
| `num_shards` | Optional target shard count. |
| `recursive` | Whether directory inputs should be scanned recursively. |
| `file_path_column` | Metadata column that records the source file path. |
| `dtypes` | Optional schema hints used by downstream writers and expressions. |

See [Sharding](sharding.md) for how these options affect worker work.

## Path inputs

Readers accept local paths and fsspec-backed URLs. Refiner commonly uses:

| Prefix | Example |
| --- | --- |
| Local path | `/data/episodes/*.hdf5` |
| Hugging Face dataset | `hf://datasets/org/name` |
| Hugging Face bucket | `hf://buckets/org/bucket/path` |
| Cloud/object storage supported by fsspec | `s3://bucket/path` |

See [Path Formats](../reference/path-formats.md).

## Related pages

- [Sharding](sharding.md)
- [Episode Rows](../episode-data/episode-rows.md)
- [Custom Readers](custom-readers.md)

