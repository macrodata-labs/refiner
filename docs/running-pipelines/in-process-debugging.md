---
title: "In-Process Debugging"
description: "Inspect Refiner pipelines directly in the current Python process"
---

# In-Process Debugging

In-process execution is the fastest way to validate a reader, inspect row shape,
or test a transform. It does not create a cloud job and does not use the local
launcher worker supervisor.

## Inspect One Row

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

## Iterate Lazily

```python
for row in pipeline.iter_rows():
    print(row.episode_id, row.num_frames)
    break
```

`iter_rows()` streams rows from the source through all transforms. It is useful
for long datasets because it does not materialize the whole pipeline.

## Materialize Small Fixtures

```python
rows = pipeline.take(5)
```

Prefer `take()` over `materialize()` unless the dataset is intentionally tiny.
`materialize()` computes every output row into memory.

## Debug A Transform

```python
def show_shape(row):
    print(row.episode_id, row.num_frames)
    return row

pipeline.map(show_shape).take(3)
```

Keep debug transforms small and remove them before launching real jobs.

## Sinks In Process

A pipeline with a sink writes when launched. For quick inspection, usually
inspect the pipeline before the sink:

```python
source = mdr.read_lerobot("hf://datasets/macrodata/aloha_static_battery_ep005_009")
trimmed = source.map(mdr.robotics.motion_trim(threshold=0.001))

print(trimmed.take(1)[0].num_frames)
```

Then attach the writer once the transform does what you expect:

```python
pipeline = trimmed.write_lerobot("hf://buckets/acme-robotics/aloha_trimmed")
```

## Related Pages

- [Episode Rows](../episode-data/episode-rows.md)
- [Row Transforms](../transforms/row-transforms.md)
- [Motion Trimming](../episode-operations/motion-trimming.md)

