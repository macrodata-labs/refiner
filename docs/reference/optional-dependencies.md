---
title: "Optional Dependencies"
description: "Install Refiner extras for specific readers and workflows"
---

# Optional Dependencies

Install extras based on the data and operations you use.

| Extra | Enables |
| --- | --- |
| `robotics` | LeRobot workflows, video dependencies, Hugging Face Hub helpers. |
| `egocentric` | Egocentric hand tracking. |
| `hdf5` | HDF5 reader support. |
| `zarr` | Zarr reader and writer support. |
| `video` | Video decode/write support. |
| `text` | Common Crawl text readers. |
| `s3` | S3 filesystem support. |
| `all` | Development/testing superset. |

Examples:

```bash
pip install macrodata-refiner[robotics]
pip install macrodata-refiner[hdf5,zarr]
pip install macrodata-refiner[egocentric]
```

