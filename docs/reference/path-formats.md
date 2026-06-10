---
title: "Path formats"
description: "Path and URL formats accepted by Refiner readers and writers"
---

# Path formats

Refiner uses fsspec-backed paths for many readers and writers.

| Format | Example |
| --- | --- |
| Local path | `/data/episodes/*.hdf5` |
| Hugging Face dataset | `hf://datasets/org/name` |
| Hugging Face bucket | `hf://buckets/org/bucket/path` |
| S3 | `s3://bucket/path` |
| Other fsspec filesystems | Depends on the installed filesystem package. |

Private remote paths usually require secrets such as `HF_TOKEN` or cloud
provider credentials. See [Secrets and environment](../platform/secrets-and-environment.md).
