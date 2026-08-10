---
title: "Hugging Face"
description: "Read Hugging Face datasets and HF-hosted files"
---

# Hugging Face

Refiner supports two common Hugging Face patterns:

1. Dataset roots and files accessed through `hf://...` paths.
2. Hugging Face datasets loaded through `read_hf_dataset`.

## HF paths

Most file-based readers can use `hf://` paths:

```python
pipeline = mdr.read_lerobot("hf://datasets/lerobot/aloha_sim_transfer_cube_human")
```

For private datasets, set `HF_TOKEN` in your environment before running the
pipeline. Local workers inherit it automatically:

```bash
export HF_TOKEN="your-read-token"
```

## Dataset tables

Use `read_hf_dataset` when you want the Hugging Face datasets library to load a
split:

```python
pipeline = mdr.read_hf_dataset(
    "acme/robot-labels",
    split="train",
)
```

Install `macrodata-refiner[datasets]` to use `read_hf_dataset(...)`.

Use this for table-style datasets. For LeRobot dataset roots, prefer
[`read_lerobot`](lerobot.md).

## Related pages

- [Path Formats](../reference/path-formats.md)
- [Local launcher](../running-pipelines/local-launcher.md)
