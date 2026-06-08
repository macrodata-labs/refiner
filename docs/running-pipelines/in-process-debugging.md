---
title: "In-process debugging"
description: "Inspect Refiner pipelines directly in the current Python process"
---

# In-process debugging

In-process execution is the fastest way to validate a reader, inspect row shape,
or test a transform. It does not create a cloud job and does not use the local
launcher worker supervisor.

## Inspect one row

```python
import refiner as mdr

pipeline = mdr.read_lerobot("hf://datasets/macrodata/aloha_static_battery_ep005_009")
row = pipeline.take(1)[0]

print(row.episode_id)
print(row.num_frames)
print(row.to_frame_table().names)
```

Use this before writing transforms. It answers the most important question:
"What keys and semantic properties does my row actually expose?"

## Iterate lazily

```python
for row in pipeline.iter_rows():
    print(row.episode_id, row.num_frames)
    break
```

`iter_rows()` streams rows from the source through all transforms. It is useful
for long datasets because it does not materialize the whole pipeline.

## Materialize small fixtures

```python
rows = pipeline.take(5)
```

Prefer `take()` over `materialize()` unless the dataset is intentionally tiny.
`materialize()` computes every output row into memory.

## Debug a transform

```python
def show_shape(row):
    print(row.episode_id, row.num_frames)
    return row

pipeline.map(show_shape).take(3)
```

Keep debug transforms small and remove them before launching real jobs.

## Sinks in process

Sinks do not write through in-process inspection methods like `take()` or
`iter_rows()`. Inspect the pipeline before attaching the sink:

```python
source = mdr.read_lerobot("hf://datasets/macrodata/aloha_static_battery_ep005_009")
trimmed = source.map(mdr.robotics.motion_trim(threshold=0.001))

print(trimmed.take(1)[0].num_frames)
```

Attach the writer when you are ready to launch the pipeline:

```python
pipeline = trimmed.write_lerobot("hf://buckets/acme-robotics/aloha_trimmed")
```

## Related pages

- [Episode Rows](../episode-data/episode-rows.md)
- [Row Transforms](../transforms/row-transforms.md)
- [Motion Trimming](../episode-operations/motion-trimming.md)
