---
title: "Reward Scoring"
description: "Score episode progress and success with a pooling model"
---

# Reward Scoring

`reward_score` samples frames from an episode video and uses a Robometer-style
pooling model to estimate progress and success. It expects rows from
`read_lerobot(...)`.

```python
pipeline = (
    mdr.read_lerobot("hf://datasets/nvidia/LIBERO_LeRobot_v3/libero_90")
    .map_async(
        mdr.robotics.reward_score(
            model="robometer/Robometer-4B",
            video_key="observation.images.image",
            task="complete the robot manipulation task",
            max_frames=8,
            max_concurrent_requests=256,
        ),
        max_in_flight=256,
        preserve_order=False,
    )
    .write_lerobot("hf://buckets/acme-robotics/libero-robometer-reward")
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

`reward_score` uses pooling inference through a vLLM provider. See
[Pooling](../inference/pooling.md) and [Providers and vLLM](../inference/providers-and-vllm.md).

If the selected episode video has no decoded frames, `reward_score` raises for
that row instead of emitting empty reward columns.

For a complete cloud example, see
[`examples/lerobot/robometer_reward.py`](../../examples/lerobot/robometer_reward.py).
