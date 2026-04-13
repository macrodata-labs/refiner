---
title: "Preparing Data for SARM"
description: "Use Refiner to annotate LeRobot episodes with model-generated SARM metadata and write the result back out"
---

The useful split for SARM is:

1. Refiner reads and rewrites the LeRobot dataset.
2. Refiner uses `map_async(...)` with `mdr.inference.generate(...)` to produce
   subtask annotations.
3. Refiner writes those annotation fields back into the episode rows.
4. After the write, Refiner emits `meta/temporal_proportions_*.json`.

This mirrors the practical contract in LeRobot's SARM annotation pipeline:

- episode-level subtask metadata is stored in `meta/episodes/*.parquet`
- dataset-level temporal priors are stored in `meta/temporal_proportions_*.json`

## Dense Annotation Example

The checked-in example lives at [examples/lerobot/sarm_annotation.py](/Users/hynky/.codex/worktrees/d544/refiner/examples/lerobot/sarm_annotation.py).

It does the following:

- reads a LeRobot dataset with `read_lerobot(...)`
- extracts one full bounded episode clip from the configured video key
- encodes that clip as a base64 `data:video/mp4` URL
- sends that video to Qwen3-VL through `mdr.inference.generate(...)` with `VLLMProvider`
- parses the returned JSON subtask segmentation
- writes `dense_subtask_*` fields onto each episode row
- auto-generates a single sparse `"task"` stage for compatibility with `dense_only`
- writes the annotated LeRobot dataset
- computes `meta/temporal_proportions_sparse.json`
- computes `meta/temporal_proportions_dense.json`

The main flow is:

```python
pipeline = (
    mdr.read_lerobot(INPUT_DATASET)
    .map_async(
        mdr.inference.generate(
            fn=annotate_dense_subtasks,
            provider=PROVIDER,
            default_generation_params={"temperature": 0.1},
            max_concurrent_requests=MAX_IN_FLIGHT,
        ),
        max_in_flight=MAX_IN_FLIGHT,
    )
    .write_lerobot(OUTPUT_DATASET)
)

pipeline.launch_cloud(name="lerobot-sarm-annotation", num_workers=1)
write_temporal_proportions(OUTPUT_DATASET, prefix="sparse")
write_temporal_proportions(OUTPUT_DATASET, prefix="dense")
```

## Output Fields

The example writes these episode-level fields:

- `dense_subtask_names`
- `dense_subtask_start_times`
- `dense_subtask_end_times`
- `dense_subtask_start_frames`
- `dense_subtask_end_frames`
- `sparse_subtask_names`
- `sparse_subtask_start_times`
- `sparse_subtask_end_times`
- `sparse_subtask_start_frames`
- `sparse_subtask_end_frames`

For sparse annotations, it also writes the legacy unprefixed aliases:

- `subtask_names`
- `subtask_start_times`
- `subtask_end_times`
- `subtask_start_frames`
- `subtask_end_frames`

## Training

For the example above, the intended LeRobot training mode is `dense_only`,
because the dense annotations are model-generated and the sparse stage is the
auto-generated single `"task"` segment:

```bash
lerobot-train \
  --dataset.repo_id=your-username/your-sarm-annotated-dataset \
  --policy.type=sarm \
  --policy.annotation_mode=dense_only \
  --policy.image_key=observation.images.main \
  --policy.state_key=observation.state \
  --output_dir=outputs/train/sarm_dense_only \
  --batch_size=32 \
  --steps=5000 \
  --policy.repo_id=your-username/your-model-name
```

## Notes

- The example is configured for Refiner-managed VLLM with
  `Qwen/Qwen3-VL-30B-A3B-Instruct`.
- The video clip is produced from the episode-local `VideoFile` bounds, so this
  works even when the source MP4 stores multiple concatenated episodes.
- If you want the full dual sparse+dense setup from LeRobot's own annotation
  script, extend the example to generate both vocabularies and write both
  temporal-proportion files.

## Internal Notes

- Refiner's LeRobot writer already preserves arbitrary episode-level row fields,
  so SARM annotation columns can be attached directly to the rows before
  `write_lerobot(...)`.
- Temporal-proportion JSON is a separate post-write step because the current
  writer finalizes LeRobot metadata files but does not own SARM-specific sidecar
  artifacts.
