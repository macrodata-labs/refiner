---
title: "Reward Scoring"
description: "Score episode progress and success with a pooling model"
---

# Reward Scoring

`reward_score` samples frames from an episode video and uses a Robometer-style
pooling model to estimate progress and success.

```python
pipeline = (
    mdr.read_lerobot("hf://datasets/acme/demos")
    .map_async(
        mdr.robotics.reward_score(
            model="robometer/Robometer-4B",
            video_key="observation.images.top",
            task=lambda row: "; ".join(row.tasks),
            max_frames=8,
        ),
        max_in_flight=32,
    )
)
```

## Output Columns

By default, the operation writes:

| Column | Meaning |
| --- | --- |
| `reward_score` | Expected progress for sampled frames. |
| `robometer_success` | Success probability for sampled frames. |

Customize names with `output_column` and `success_column`.

## Task Text

Task text can come from:

```python
mdr.robotics.reward_score(task="open the drawer")
```

or from the row:

```python
mdr.robotics.reward_score(task=lambda row: row.task or "; ".join(row.tasks))
```

## Inference Backend

`reward_score` uses pooling inference through a VLLM provider. See
[Pooling](../inference/pooling.md) and [Providers and VLLM](../inference/providers-and-vllm.md).
