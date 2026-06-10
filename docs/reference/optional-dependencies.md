---
title: "Optional dependencies"
description: "Install Refiner extras for specific readers and workflows"
---

# Optional dependencies

Install extras based on the data and operations you use.

| Extra | Enables |
| --- | --- |
| `all` | Includes every extra below; useful for testing and development. |
| `datasets` | Hugging Face dataset reader support; includes `hf`. |
| `hf` | Hugging Face Hub APIs and HF filesystem helpers. |
| `hand_tracking` | Hand tracking with ego-vision. |
| `hdf5` | HDF5 reader support. |
| `zarr` | Zarr reader and writer support. |
| `mcap` | MCAP robotics log reader support, including ROS2, protobuf, and H.264 video decoding. |
| `video` | Video decode/write support. |
| `text` | Common Crawl text readers. |
| `s3` | S3 filesystem support. |
| `gcs` | Google Cloud Storage filesystem support. |
| `tensorflow` | Tensorflow support. |
| `tfds` | Tensorflow datasets support. |

Examples:

```bash
pip install macrodata-refiner[hf,video]
pip install macrodata-refiner[datasets]
pip install macrodata-refiner[hdf5,zarr]
pip install macrodata-refiner[mcap]
pip install macrodata-refiner[hand_tracking]
```
