# LeRobot Benchmarks

This folder contains benchmark harnesses for LeRobot workloads:

- `run_merge_benchmark.py`: official `lerobot` merge path vs Refiner `read_lerobot(...).write_lerobot(...)`
- `run_delete_benchmark.py`: official `lerobot` episode deletion vs Refiner `read_lerobot(...).flat_map(...).write_lerobot(...)`

The merge benchmark uses the two 5-episode Hub datasets created from `lerobot/aloha_static_battery`:

- `macrodata/aloha_static_battery_ep000_004`
- `macrodata/aloha_static_battery_ep005_009`

Each benchmark case runs with a fresh isolated Hugging Face cache directory. The runner deletes that cache directory before the case starts, so the official `lerobot` path has to download the source datasets again.

The two implementations start from the Hub differently:

- official `lerobot`: `LeRobotDataset(repo_id=..., download_videos=True)` with an empty isolated HF cache
- Refiner: `read_lerobot("hf://datasets/...")` with the same empty-cache setup

The official merge API does not accept a true `hf://datasets/...` root. Passing `hf://...` there is treated as a local filesystem path, so this benchmark uses the closest valid cold-start configuration for the official side.

## Prerequisites

- `lerobot` must be installed in the active environment.
- Hugging Face auth must be available if the source datasets are private.

The delete benchmark uses one input dataset by default:

- `macrodata/aloha_static_battery_ep000_004`

and deletes episode `1` by default.

## Run

```bash
uv run python benchmark/lerobot/run_merge_benchmark.py
```

```bash
uv run python benchmark/lerobot/run_delete_benchmark.py
```

Useful options:

- `--iterations 3` to repeat cold runs
- `--mode official` or `--mode refiner` to run one implementation only
- `--artifacts-dir /tmp/refiner-lerobot-benchmark` to write outputs elsewhere
- `--num-workers 1` to control Refiner local launcher workers
- `--upload-hf-bucket hf://buckets/macrodata/merge-benchmark` to upload outputs to a Hugging Face bucket instead of keeping them only under the local artifacts tree
- `--delete-episode 1 --delete-episode 3` to benchmark deletion of specific episode indices

For merge and delete uploads, `--upload-hf-bucket` accepts either:

- `owner/bucket[/prefix]`
- `hf://buckets/owner/bucket[/prefix]`

Each benchmark derives one bucket subdirectory per implementation and iteration:

- official: local merge, then `sync_bucket(...)` to `hf://buckets/owner/bucket[/prefix]/<timestamp>/official/iteration-XX`
- refiner: direct `write_lerobot("hf://buckets/owner/bucket[/prefix]/<timestamp>/refiner/iteration-XX")`
- delete benchmark outputs are written under `hf://buckets/owner/bucket[/prefix]/delete/<timestamp>/<implementation>/iteration-XX`

Artifacts are written under `benchmark/lerobot/artifacts/` by default:

- per-case result JSON
- merged output directories
- isolated Hugging Face cache directories
- one summary JSON for the whole benchmark session
