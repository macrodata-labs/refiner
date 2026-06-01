import refiner as mdr

RAW_DATASETS = [
    "hf://datasets/openvla/modified_libero_rlds/libero_10_no_noops/1.0.0",
    "hf://datasets/openvla/modified_libero_rlds/libero_goal_no_noops/1.0.0",
    "hf://datasets/openvla/modified_libero_rlds/libero_object_no_noops/1.0.0",
    "hf://datasets/openvla/modified_libero_rlds/libero_spatial_no_noops/1.0.0",
]

pipeline = (
    mdr.read_tfds(
        RAW_DATASETS,
        videos={
            "front": "steps/observation/image",
            "wrist": "steps/observation/wrist_image",
        },
        fps=10,
    )
    .to_robot_rows(
        nested_frames_key="steps",
        episode_id_key="episode_metadata/file_path",
        task_key="steps/language_instruction",
        action_key="action",
        state_key="observation/state",
        video_keys={
            "observation.images.front": "videos/front",
            "observation.images.wrist": "videos/wrist",
        },
        fps=10,
        robot_type="libero",
    )
    .write_lerobot("./libero-10-lerobot")
)

pipeline.launch_local(
    name="libero_rlds_to_lerobot",
    num_workers=4,
)
