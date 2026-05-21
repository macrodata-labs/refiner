from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import refiner as mdr


RAW_INPUT = os.environ["REFINER_EGO_INPUT"]
HF_OUTPUT = os.environ["REFINER_EGO_OUTPUT"]
HAWOR_EXPORT = os.environ.get("REFINER_HAWOR_EXPORT", "/opt/HaWoR/refiner_export.py")
HAWOR_SETUP = os.environ.get("REFINER_HAWOR_SETUP")
CLIP_SECONDS = float(os.environ.get("REFINER_EGO_CLIP_SECONDS", "0"))
_HAWOR_READY = False
_SCRIPT_DIR = Path(__file__).resolve().parent
_BUNDLED_RUNTIME_FILES = {
    "install_hawor_runtime.sh": (_SCRIPT_DIR / "install_hawor_runtime.sh").read_text(),
    "patch_hawor_safetensors_native.py": (
        _SCRIPT_DIR / "patch_hawor_safetensors_native.py"
    ).read_text(),
    "hawor_refiner_export.py": (_SCRIPT_DIR / "hawor_refiner_export.py").read_text(),
}


def write_bundled_runtime() -> Path:
    runtime_dir = Path("/tmp/refiner-hawor-runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for name, contents in _BUNDLED_RUNTIME_FILES.items():
        path = runtime_dir / name
        path.write_text(contents)
        if name.endswith(".sh"):
            path.chmod(0o755)
    return runtime_dir


def ensure_hawor_runtime():
    global _HAWOR_READY
    if _HAWOR_READY:
        return
    if not Path(HAWOR_EXPORT).exists() or not _hawor_renderer_available():
        runtime_dir = write_bundled_runtime()
        setup = HAWOR_SETUP
        if not setup:
            setup = (
                f"REFINER_EXPORT={runtime_dir / 'hawor_refiner_export.py'} "
                f"REFINER_PATCH_SAFETENSORS={runtime_dir / 'patch_hawor_safetensors_native.py'} "
                f"bash {runtime_dir / 'install_hawor_runtime.sh'}"
            )
        completed = subprocess.run(
            setup,
            shell=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"HaWoR setup failed with exit code {completed.returncode}"
            )
    if not _hawor_renderer_available():
        raise RuntimeError(
            "HaWoR renderer is unavailable after setup; PyTorch3D is required "
            "for official HaWoR hand masks."
        )
    _HAWOR_READY = True


def _hawor_renderer_available() -> bool:
    hawor_root = Path(HAWOR_EXPORT).resolve().parent
    if not hawor_root.exists():
        return False
    completed = subprocess.run(
        [
            "python",
            "-c",
            "import pytorch3d; from lib.vis.renderer import Renderer",
        ],
        cwd=hawor_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def reconstruct_hawor(row):
    ensure_hawor_runtime()
    video_path = _materialize_video_path(str(row["file_path"]))
    output_dir = Path("/tmp/hawor-artifacts") / "sample"
    output_dir.mkdir(parents=True, exist_ok=True)
    if CLIP_SECONDS > 0:
        clip_path = output_dir / "input_clip.mp4"
        _write_clip(video_path, clip_path, CLIP_SECONDS)
        video_path = str(clip_path)
    result_path = output_dir / "hawor_result.json"
    completed = subprocess.run(
        [
            "python",
            HAWOR_EXPORT,
            "--video",
            video_path,
            "--result",
            str(result_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    with result_path.open("r", encoding="utf-8") as raw:
        hawor = json.load(raw)
    return row.update({"hawor": hawor})


def _materialize_video_path(video_path: str) -> str:
    if not video_path.startswith("hf://datasets/"):
        return video_path
    from huggingface_hub import hf_hub_download

    rest = video_path.removeprefix("hf://datasets/")
    repo_parts = rest.split("/", 2)
    if len(repo_parts) != 3:
        raise RuntimeError(f"unsupported Hugging Face dataset URI: {video_path}")
    namespace, name, filename = repo_parts
    return hf_hub_download(
        repo_id=f"{namespace}/{name}",
        repo_type="dataset",
        filename=filename,
        token=os.environ.get("HF_TOKEN"),
    )


def _write_clip(input_path: str, output_path: Path, seconds: float) -> None:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            input_path,
            "-t",
            str(seconds),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "video clipping failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    if output_path.stat().st_size == 0:
        raise RuntimeError("video clipping failed: empty output")


def make_actions(row):
    hawor = row["hawor"]
    timestamps = list(hawor["timestamps"])
    out = {
        "timestamps": timestamps[:-1],
        "target_timestamps": timestamps[1:],
        "horizon": 1,
        "hands": {},
    }
    for side in ("left", "right"):
        hand = hawor.get(f"{side}_hand")
        if not hand or "T_world_wrist" not in hand:
            continue
        wrist = np.asarray(hand["T_world_wrist"], dtype=np.float64)
        out["hands"][side] = {
            "wrist_delta": [
                (np.linalg.inv(wrist[idx]) @ wrist[idx + 1]).tolist()
                for idx in range(len(timestamps) - 1)
            ]
        }
        if "mano_pose" in hand:
            out["hands"][side]["mano_target"] = hand["mano_pose"][1:]
        if "confidence" in hand:
            out["hands"][side]["confidence"] = hand["confidence"][:-1]
    return row.update({"ego_actions": out})


pipeline = (
    mdr.read_files(RAW_INPUT)
    .map(reconstruct_hawor)
    .map(make_actions)
    .write_jsonl(HF_OUTPUT)
)


if __name__ == "__main__":
    pipeline.launch_cloud(
        name=os.environ.get("REFINER_JOB_NAME", "hawor-egocentric-inline"),
        num_workers=int(os.environ.get("REFINER_NUM_WORKERS", "1")),
        cpus_per_worker=int(os.environ.get("REFINER_CPUS_PER_WORKER", "8")),
        mem_mb_per_worker=int(os.environ.get("REFINER_MEM_MB_PER_WORKER", "32768")),
        gpu=mdr.GPU(count=1, type="h100", cuda_version="12.4"),
        secrets=mdr.Secrets.env(name="default", keys=["HF_TOKEN"]),
        env={"MACRODATA_BASE_URL": "https://dev.macrodata.co"},
    )
