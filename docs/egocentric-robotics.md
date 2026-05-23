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

## VLA Action Benchmark

For robot foundation model training, benchmark the relative action signal rather
than only absolute 3D hand pose. Refiner scores a prediction against a ground
truth HaWoR-style payload by converting both sequences into wrist-frame deltas:

```python
import refiner as mdr

metrics = mdr.robotics.egocentric.score_vla_relative_actions(
    predicted_hawor_payload,
    hot3d_ground_truth_payload,
    horizon=1,
    hands=("left", "right"),
    confidence_threshold=0.5,
)
```

For HOT3D train/validation tar files, load the target wrist trajectory directly:

```python
target = mdr.robotics.egocentric.load_hot3d_tar_ground_truth(
    "train_aria/clip-001849.tar",
    stream_id="214-1",
    fps=30.0,
)
```

The primary action is:

```python
action[t] = inverse(T_world_wrist[t]) @ T_world_wrist[t + horizon]
```

The benchmark reports:

- wrist delta translation error in meters
- wrist delta rotation error in degrees
- MANO pose delta error when `mano_pose` is available
- prediction coverage after confidence gating
- predicted action jitter
- absolute wrist translation error as a diagnostic only

Use this benchmark shape for HOT3D hillclimbing. HOT3D ground-truth camera and
hand annotations are useful for scoring, but production pipelines should still
be evaluated in tiers: RGB-only, RGB plus known intrinsics, RGB plus learned
metric depth, and oracle HOT3D calibration for an upper bound.

## Temporal Filter Scoring

Use `examples/egocentric/hot3d_temporal_filter_scoring.py` to evaluate
temporal filters on saved HOT3D HaWoR outputs without running HaWoR again. The
script reads normalized HaWoR JSON predictions and HOT3D annotation tar files,
applies prediction-only filters, then calls `score_vla_relative_actions(...)`
for the final metrics.

```bash
uv run python examples/egocentric/hot3d_temporal_filter_scoring.py \
  --source-root /cache/hot3d/benchmarks/focal-clean-10 \
  --ground-truth-root /cache/hot3d/train_aria
```

For explicit path control, pass a JSONL manifest with `clip`, `experiment`,
`prediction_path`, and `ground_truth_path` fields. The script writes
`*-scores.jsonl`, `*-summary.json`, and `*-logbook.md` artifacts. Filters must
not read HOT3D ground truth; ground truth is only loaded after filtering for
scoring.

## Composable Geometry Pipeline

AoE-style preprocessing should be represented as separate geometry stages, not
as one monolithic backend:

```python
import refiner as mdr

pipeline = mdr.robotics.egocentric.make_aoe_like_pipeline(
    depth=LingBotDepthEstimator(),
    camera=MegaSAMTrajectoryEstimator(),
    hands=HaWoRHandReconstructor(),
    projector=mdr.robotics.egocentric.HandWorldProjector(),
)
```

The intended stage contract is:

```text
DepthEstimator
-> CameraTrajectoryEstimator
-> HandReconstructor
-> HandWorldProjector
```

For an AoE-like recipe, the modules map to:

- `DepthEstimator`: LingBot-Depth or another depth-prior backend.
- `CameraTrajectoryEstimator`: MegaSAM, producing `camera.T_world_camera`.
- `HandReconstructor`: HaWoR, producing camera-space MANO pose, wrist pose,
  joints, and confidence.
- `HandWorldProjector`: Refiner projection, preserving MANO fields while
  computing world-space wrist transforms and joints from the selected camera
  trajectory.

Refiner exposes `estimate_depth_lingbot(...)` for depth-stage integrations that
run an external LingBot-Depth command and return a normalized depth payload.
For the MegaSAM runtime, the AoE-like recipe uses Depth-Anything for monocular
inverse depth, UniDepth for raw metric depth, LingBot-Depth to refine that raw
metric depth, then MegaSAM consumes the refined per-frame `.npz` depth stream.

This separation lets the same HaWoR hand reconstruction be fused with either
HaWoR's own camera trajectory or an external MegaSAM trajectory:

```text
HaWoR official:
  HaWoR depth -> HaWoR/DROID trajectory -> HaWoR hands -> HaWoR/Refiner world hands

AoE-like:
  LingBot-Depth -> MegaSAM trajectory -> HaWoR hands -> Refiner world hands

VGGT-Omega:
  VGGT-Omega geometry -> HaWoR hands -> Refiner world hands
```

VGGT-Omega is represented as a combined geometry backend because the model
predicts camera parameters and depth in one forward pass. Use
`estimate_geometry_vggt_omega(...)` to run an external official VGGT-Omega
runtime and load the normalized output into both camera and depth columns:

```python
import refiner as mdr

mapper = mdr.robotics.egocentric.estimate_geometry_vggt_omega(
    command=[
        "python",
        "examples/egocentric/vggt_omega_refiner_export.py",
        "--video",
        "{video_path}",
        "--output-dir",
        "{output_dir}",
        "--result",
        "{result_path}",
        "--checkpoint",
        "/checkpoints/VGGT-Omega-1B-512/model.pt",
        "--repo-path",
        "/opt/vggt-omega",
    ],
)
```

The resulting row contains:

- `vggt_omega`: the full normalized geometry payload.
- `vggt_omega_camera`: Refiner camera payload with `T_world_camera`.
- `vggt_omega_depth`: Refiner depth payload pointing at the raw VGGT-Omega
  `.npz` artifact.

The official VGGT-Omega API returns camera extrinsics plus intrinsics and depth.
Refiner stores camera motion as `T_world_camera`, so the adapter treats
VGGT-style extrinsics as world-to-camera by default and inverts them. If a raw
artifact already stores camera-to-world transforms, convert it with
`write_vggt_omega_geometry_json(..., extrinsic_convention="camera_to_world")`.

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
