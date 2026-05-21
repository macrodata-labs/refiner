---
title: "Egocentric Robotics"
description: "Extracting hand actions from egocentric video"
---

Refiner can orchestrate egocentric hand-action extraction by treating external
reconstruction systems, such as HaWoR, as pipeline operators.

The first supported contract is HaWoR-style world-space hand reconstruction:

```python
import refiner as mdr

pipeline = (
    mdr.read_files("data/ego/*.mp4")
    .map(
        mdr.robotics.egocentric.reconstruct_hands_hawor(
            command=[
                "python",
                "/opt/HaWoR/refiner_export.py",
                "--video",
                "{video_path}",
                "--result",
                "{result_path}",
            ],
            output_root="artifacts/hawor",
        )
    )
    .map(mdr.robotics.egocentric.make_relative_actions(hands=("left", "right")))
    .map(
        mdr.robotics.egocentric.export_rerun(
            output_root="artifacts/rerun",
        )
    )
    .write_jsonl("out/ego-actions")
)
```

Refiner does not vendor HaWoR or depend on its research runtime. HaWoR requires
CUDA, DROID-SLAM, Metric3D, public HaWoR/WiLoR checkpoints, and separately
licensed MANO model files. Install and run those components outside Refiner,
then use Refiner to shard videos, run the command, validate outputs, compute
relative actions, and write training datasets.

Refiner includes a repo-owned HaWoR adapter and setup helper:

- [`examples/egocentric/hawor_refiner_export.py`](../examples/egocentric/hawor_refiner_export.py)
- [`examples/egocentric/install_hawor_runtime.sh`](../examples/egocentric/install_hawor_runtime.sh)

The setup helper clones the official HaWoR repository, installs Python
dependencies, installs masked DROID-SLAM, downloads the native HaWoR/WiLoR
weights from Hugging Face, downloads DROID-SLAM, Metric3D, and MANO safetensors
assets from `macrodata/hawor-safetensors`, then patches HaWoR/DROID/Metric3D
loaders to read those safetensors directly. It also copies
`hawor_refiner_export.py` to `/opt/HaWoR/refiner_export.py`.
The helper keeps HaWoR's PyTorch3D renderer path enabled so the official
MANO-mesh hand masks are used for masked DROID-SLAM. If PyTorch3D or
`lib.vis.renderer.Renderer` cannot be imported, setup fails instead of falling
back to bounding-box masks.

HaWoR itself is licensed as CC-BY-NC-ND, and MANO is separately licensed. For
that reason, Refiner does not copy HaWoR source code or MANO files into the
package.

## HaWoR Output Contract

The command passed to `reconstruct_hands_hawor(...)` must write a JSON file at
`{result_path}`. The normalized shape is:

```json
{
  "timestamps": [0.0, 0.033],
  "camera": {
    "T_world_camera": [
      [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      [[1, 0, 0, 0.01], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    ]
  },
  "right_hand": {
    "T_world_wrist": [
      [[1, 0, 0, 0.3], [0, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0, 1]],
      [[1, 0, 0, 0.32], [0, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0, 1]]
    ],
    "mano_pose": [[0.0], [0.1]],
    "confidence": [0.92, 0.9]
  }
}
```

`left_hand` has the same shape as `right_hand`. Hand fields are optional, but
every sequence field present must have the same length as `timestamps`.

## Cloud Run To HF

Use Macrodata Cloud for the HaWoR inference stage because HaWoR depends on a
CUDA research stack. Mount `HF_TOKEN` from a saved workspace secret environment:

```python
import refiner as mdr

pipeline = (
    mdr.read_files("hf://datasets/your-org/raw-ego-videos/**/*.mp4")
    .map(
        mdr.robotics.egocentric.reconstruct_hands_hawor(
            command=[
                "python",
                "/opt/HaWoR/refiner_export.py",
                "--video",
                "{video_path}",
                "--result",
                "{result_path}",
            ],
            output_root="/tmp/hawor-artifacts",
        )
    )
    .map(mdr.robotics.egocentric.make_relative_actions())
    .write_jsonl("hf://datasets/your-org/ego-hawor-actions")
)

result = pipeline.launch_cloud(
    name="hawor-egocentric-actions",
    num_workers=8,
    gpu=mdr.GPU(count=1, type="h100", cuda_version="12.4"),
    secrets=mdr.Secrets.env(name="default", keys=["HF_TOKEN"]),
    env={"MACRODATA_BASE_URL": "https://dev.macrodata.co"},
)
```

The same launch shape is available as
[`examples/egocentric/hawor_cloud_to_hf.py`](../examples/egocentric/hawor_cloud_to_hf.py).
Set `MACRODATA_BASE_URL=https://dev.macrodata.co` when authenticating or
submitting to the dev control plane.

For a one-off dev job before the new Refiner module is published, use the
inline example:

```bash
export MACRODATA_BASE_URL=https://dev.macrodata.co
export REFINER_EGO_INPUT='hf://datasets/yixuan-tan/EgoDex-LeRobot-v3.0/test/basic_pick_place/videos/observation.images.camera/chunk-000/file-000.mp4'
export REFINER_EGO_OUTPUT='hf://datasets/yixuan-tan/egodex-hawor-debug'
export REFINER_HAWOR_SETUP='/path/to/install_hawor_runtime.sh'

uv run python examples/egocentric/hawor_cloud_inline.py
```

`REFINER_HAWOR_SETUP` must be a command visible inside the cloud worker. The
setup command uses `HAWOR_SAFETENSORS_REPO=macrodata/hawor-safetensors` by
default. The setup patches the cloned HaWoR runtime so DROID-SLAM, Metric3D,
and MANO load from `weights/safetensors/*.safetensors` directly. No MANO
`.pkl` is required by the default path. The setup also verifies that HaWoR's
PyTorch3D renderer imports successfully; this is required for the official
hand-mask generation used by masked DROID-SLAM.

## Relative Actions

`make_relative_actions(...)` computes wrist deltas from world-space wrist poses:

```python
delta = inverse(T_world_wrist[t]) @ T_world_wrist[t + horizon]
```

The output is a nested row column named `ego_actions` by default:

```json
{
  "timestamps": [0.0],
  "target_timestamps": [0.033],
  "horizon": 1,
  "hands": {
    "right": {
      "wrist_delta": [
        [[1, 0, 0, 0.02], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
      ],
      "mano_target": [[0.1]],
      "confidence": [0.92]
    }
  }
}
```

Use this as an intermediate action representation. Robot-specific conversion,
such as retargeting human keypoints to a dexterous robot hand, should be a later
pipeline step.

## Rerun Visualization

Install the optional egocentric dependencies in the environment that exports
visualizations:

```bash
uv sync --extra egocentric
```

Export a saved HaWoR artifact to a Rerun recording:

```python
import refiner as mdr

row = mdr.read_jsonl("hf://datasets/your-org/ego-hawor-actions/*.jsonl").take(1)[0]

mdr.robotics.egocentric.export_hawor_rerun(
    mdr.robotics.egocentric.HaworResult.from_mapping(row["hawor"]),
    output_path="debug/hawor.rrd",
)
```

Open the recording:

```bash
rerun debug/hawor.rrd
```

The exporter logs camera trajectories, wrist trajectories, current wrist points,
hand keypoints when present, hand skeletons for 21-joint hands, and confidence
time series.

## Internal Notes

HaWoR is a provider, not a Refiner dependency. The stable Refiner boundary is
the normalized JSON artifact containing timestamps, camera poses, wrist poses,
MANO parameters, and confidence values.
