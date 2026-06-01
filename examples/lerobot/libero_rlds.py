import refiner as mdr

pipeline = (
    mdr.read_tfds(
        "hf://datasets/openvla/modified_libero_rlds/libero_10_no_noops/1.0.0",
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
