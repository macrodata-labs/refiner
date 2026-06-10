---
title: "Cloud GPU Job"
description: "Request cloud GPUs for model-heavy Refiner jobs"
---

# Cloud GPU Job
Many data workloads require running models on GPUs—whether for vision models, encoder models, or other machine learning tasks. Refiner natively supports these GPU-powered workloads!

```python
import subprocess

import refiner as mdr
from refiner.worker.context import logger


def log_nvidia_smi(task_rank: int, num_tasks: int) -> int:
    logger.info("{}", subprocess.check_output(["nvidia-smi"], text=True).rstrip())
    # Some GPU heavy job (running BERT, ViT or anything you need)
    return task_rank


pipeline = mdr.task(log_nvidia_smi, num_tasks=1)

pipeline.launch_cloud(
    name="gpu-ranks",
    num_workers=1,
    gpu=mdr.GPU(type="h100", count=1),
)
```

This example requests one H100 GPU for a single cloud worker and logs the raw output of `nvidia-smi` to the worker logs.
The scalar return value is emitted as the task row's `result` field.

For real GPU workloads, use `gpu=mdr.GPU(...)` to specify the required GPU type and count based on your model's needs. Increase the number of tasks if you need to improve parallelization.

Related: [Resources, GPUs, and Services](../../running-pipelines/resources-gpus-and-services.md).
