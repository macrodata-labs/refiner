---
title: "Reward Scoring"
description: "Score episode progress and success with a pooling model"
---

# Reward Scoring

`reward_score` samples frames from an episode video and uses a Robometer-style
pooling model to estimate progress and success.

As of right now, Refiner supports the state-of-the-art Robometer reward model
([paper](https://arxiv.org/abs/2603.02115)) for dense progress and success
signals on robot episodes.

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
| `reward_frames` | Sampled frame metadata. Each item contains the video frame `index` and `timestamp_s` used for the matching score. |

Customize names with `output_column`, `success_column`, and `frames_column`.

## Task Text

Beyond the video to score, you must provide a task description. This can be a
string or a function that takes the row and returns the task description:

```python
mdr.robotics.reward_score(task="open the drawer")
```

or from the row:

```python
mdr.robotics.reward_score(task=lambda row: row.task or "; ".join(row.tasks))
```

## Inference Backend

`reward_score` uses pooling inference through a vLLM provider. See
[Pooling](../inference/pooling.md) and [Providers and vLLM](../inference/inference_providers.md).

For a complete cloud example, see
[Reward modeling example](../examples/annotations/reward-models.md).
