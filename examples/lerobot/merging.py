import refiner as mdr

mdr.read_lerobot(
    [
        "hf://datasets/macrodata/aloha_static_battery_ep005_009",
        "hf://datasets/macrodata/aloha_static_battery_ep000_004",
    ]
).write_lerobot(
    "hf://buckets/macrodata/test_bucket/aloha_merge",
).launch_local(name="merge_aloha", num_workers=1)
