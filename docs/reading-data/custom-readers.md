---
title: "Custom Readers"
description: "Create custom Refiner sources for data systems not covered by built-in readers"
---

# Custom Readers

Use a custom source when your data cannot be described by a built-in reader.
Custom sources are most useful for internal databases, generated tasks, or
special storage layouts.

## Minimal Source

```python
from collections.abc import Iterator

import refiner as mdr
from refiner.pipeline.sources import BaseSource
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.shard import Shard


class EpisodesSource(BaseSource):
    name = "episodes"

    def list_shards(self) -> list[Shard]:
        return [Shard.from_row_range(start=0, end=10)]

    def read_shard(self, shard: Shard) -> Iterator[DictRow]:
        for episode_id in range(shard.descriptor.start, shard.descriptor.end):
            yield DictRow({"episode_id": str(episode_id), "frames": []})


pipeline = mdr.from_source(EpisodesSource())
```

The exact shard descriptor you use depends on how your source can be divided.

## Keep Planning Cheap

`list_shards()` should not download the dataset or decode media. It should do
only the work needed to create units that workers can read independently.

## Describe The Source

If your source appears in cloud job plans, implement a useful `describe()`
method:

```python
def describe(self) -> dict[str, object]:
    return {"path": "internal://episodes", "count": 10}
```

Descriptions should help users understand the job. Do not include secrets.

## Related Pages

- [Reader Model](reader-model.md)
- [Sharding](sharding.md)
- [Converting to Robot Rows](../episode-data/converting-to-robot-rows.md)
