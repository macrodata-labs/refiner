from huggingface_hub import snapshot_download

import refiner as mdr

dataset_root = snapshot_download(
    repo_id="openvla/modified_libero_rlds",
    repo_type="dataset",
    allow_patterns="libero_10_no_noops/1.0.0/*",
)
builder_dir = f"{dataset_root}/libero_10_no_noops/1.0.0"

pipeline = (
    mdr.read_tfds(
        builder_dir=builder_dir,
        videos={
            "front": "steps/observation/image",
            "wrist": "steps/observation/wrist_image",
        },
        fps=30,
    )
    .to_robot_rows(
        nested_frames_key="steps",
        action_key="action",
        state_key="observation/state",
        video_keys={
            "observation.images.front": "videos/front",
            "observation.images.wrist": "videos/wrist",
        },
        fps=30,
        robot_type="libero",
    )
    .write_lerobot("./libero-10-lerobot")
)

pipeline.launch_local(
    name="libero_rlds_to_lerobot",
    num_workers=4,
)
