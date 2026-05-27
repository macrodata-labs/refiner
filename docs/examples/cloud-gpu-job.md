---
title: "Cloud GPU Job"
description: "Request cloud GPUs for model-heavy Refiner jobs"
---

# Cloud GPU Job

```python
import refiner as mdr

pipeline.launch_cloud(
    name="vlm-annotation",
    num_workers=4,
    gpu=mdr.GPU(type="h100", count=1),
    mem_mb_per_worker=32768,
    secrets={"HF_TOKEN": None},
)
```

Use GPUs for model work that runs inside workers or attached runtime services.
For API-only workloads, tune concurrency before adding GPUs.

Related: [Resources, GPUs, and Services](../running-pipelines/resources-gpus-and-services.md).
