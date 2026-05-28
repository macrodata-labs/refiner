---
title: "Annotate Subtasks"
description: "Add VLM-generated temporal subtask annotations to episodes"
---

# Annotate Subtasks

```python
import refiner as mdr

provider = mdr.inference.VLLMProvider(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
)

pipeline = (
    mdr.read_lerobot("hf://datasets/acme/raw-demos")
    .map_async(
        mdr.robotics.subtask_annotation(
            provider=provider,
            video_key="observation.images.top",
            output_column="predicted_subtasks",
        ),
        max_in_flight=64,
    )
    .write_lerobot("hf://buckets/acme-robotics/demos-with-subtasks")
)
```

Use [Subtask Annotation](../episode-operations/subtask-annotation.md) for
parameter details and [Providers and VLLM](../inference/providers-and-vllm.md)
for model setup.
