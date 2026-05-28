---
title: "Quickstart"
description: "Run a complete robotics data pipeline with Refiner"
---

```python
import refiner as mdr

pipeline = mdr.from_items(
    [{"text": "hello world", "lang": "en"}]
)
pipeline = pipeline.filter(mdr.col("lang") == "en")
pipeline = pipeline.write_jsonl("s3://my-bucket/example-output/")

pipeline.launch_local()
```

See [Quickstart](quickstart.md) for the full walkthrough.
