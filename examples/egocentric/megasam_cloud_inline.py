from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import refiner as mdr


RAW_INPUT = os.environ["REFINER_EGO_INPUT"]
HF_OUTPUT = os.environ["REFINER_EGO_OUTPUT"]
HF_ARTIFACT_REPO = os.environ.get("REFINER_EGO_ARTIFACT_REPO")
HF_ARTIFACT_PATH = os.environ.get(
    "REFINER_EGO_ARTIFACT_PATH", "megasam_trajectory.json"
)
MEGASAM_EXPORT = os.environ.get(
    "REFINER_MEGASAM_EXPORT", "/opt/mega-sam/refiner_export.py"
)
MEGASAM_SETUP = os.environ.get("REFINER_MEGASAM_SETUP")
CLIP_SECONDS = float(os.environ.get("REFINER_EGO_CLIP_SECONDS", "0"))
MAX_FRAMES = int(os.environ.get("REFINER_MEGASAM_MAX_FRAMES", "0"))
STRIDE = int(os.environ.get("REFINER_MEGASAM_STRIDE", "1"))
METRIC_DEPTH_BACKEND = os.environ.get("REFINER_METRIC_DEPTH_BACKEND", "unidepth")
IMG_FOCAL = os.environ.get("REFINER_IMG_FOCAL")
LINGBOT_MODEL = os.environ.get(
    "REFINER_LINGBOT_DEPTH_MODEL",
    "robbyant/lingbot-depth-pretrain-vitl-14-v0.5",
)
_MEGASAM_READY = False
_SCRIPT_DIR = Path(__file__).resolve().parent
_BUNDLED_RUNTIME_FILES = {
    "install_megasam_runtime.sh": (
        _SCRIPT_DIR / "install_megasam_runtime.sh"
    ).read_text(),
    "megasam_refiner_export.py": (
        _SCRIPT_DIR / "megasam_refiner_export.py"
    ).read_text(),
}


def write_bundled_runtime() -> Path:
    runtime_dir = Path("/tmp/refiner-megasam-runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for name, contents in _BUNDLED_RUNTIME_FILES.items():
        path = runtime_dir / name
        path.write_text(contents)
        if name.endswith(".sh"):
            path.chmod(0o755)
    return runtime_dir


def ensure_megasam_runtime() -> None:
    global _MEGASAM_READY
    if _MEGASAM_READY:
        return
    if not Path(MEGASAM_EXPORT).exists():
        runtime_dir = write_bundled_runtime()
        setup = MEGASAM_SETUP
        if not setup:
            setup = (
                f"REFINER_EXPORT={runtime_dir / 'megasam_refiner_export.py'} "
                f"bash {runtime_dir / 'install_megasam_runtime.sh'}"
            )
        print(f"[refiner-megasam] running setup: {setup}", flush=True)
        completed = subprocess.run(
            setup,
            shell=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"MegaSAM setup failed with exit code {completed.returncode}"
            )
        print("[refiner-megasam] setup complete", flush=True)
    _MEGASAM_READY = True


def estimate_megasam(row):
    ensure_megasam_runtime()
    img_focal = os.environ.get("REFINER_IMG_FOCAL") or IMG_FOCAL
    artifact_repo = os.environ.get("REFINER_EGO_ARTIFACT_REPO") or HF_ARTIFACT_REPO
    artifact_path = os.environ.get("REFINER_EGO_ARTIFACT_PATH") or HF_ARTIFACT_PATH
    video_path = _materialize_video_path(str(row["file_path"]))
    output_dir = Path("/tmp/megasam-artifacts") / "sample"
    output_dir.mkdir(parents=True, exist_ok=True)
    if CLIP_SECONDS > 0:
        clip_path = output_dir / "input_clip.mp4"
        print(f"[refiner-megasam] clipping input to {CLIP_SECONDS}s", flush=True)
        _write_clip(video_path, clip_path, CLIP_SECONDS)
        video_path = str(clip_path)
    result_path = output_dir / "megasam_trajectory.json"
    args = [
        "python",
        MEGASAM_EXPORT,
        "--video",
        video_path,
        "--result",
        str(result_path),
        "--work-dir",
        str(output_dir / "work"),
        "--scene-name",
        "refiner_scene",
        "--metric-depth-backend",
        METRIC_DEPTH_BACKEND,
        "--lingbot-model",
        LINGBOT_MODEL,
    ]
    if MAX_FRAMES > 0:
        args.extend(["--max-frames", str(MAX_FRAMES)])
    if STRIDE != 1:
        args.extend(["--stride", str(STRIDE)])
    if img_focal:
        args.extend(["--focal", img_focal])
    print(f"[refiner-megasam] running export: {' '.join(args)}", flush=True)
    completed = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "MegaSAM export failed with exit code "
            f"{completed.returncode}\n{_tail_output(completed.stdout, completed.stderr)}"
        )
    with result_path.open("r", encoding="utf-8") as raw:
        camera = json.load(raw)
    depth = camera.pop("depth", None)
    print(
        "[refiner-megasam] export complete: "
        f"{len(camera.get('T_world_camera', []))} camera poses",
        flush=True,
    )
    update = {"megasam_camera": camera}
    if depth is not None:
        update["depth"] = depth
    if artifact_repo:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        print(
            f"[refiner-megasam] uploading artifact to {artifact_repo}/{artifact_path}",
            flush=True,
        )
        uploaded = api.upload_file(
            path_or_fileobj=str(result_path),
            path_in_repo=artifact_path,
            repo_id=artifact_repo,
            repo_type="dataset",
        )
        update["megasam_artifact_url"] = uploaded
    return row.update(update)


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


def _tail_output(
    stdout: str | None, stderr: str | None, *, max_lines: int = 160
) -> str:
    lines = []
    if stdout:
        lines.extend(["--- stdout tail ---", *stdout.splitlines()[-max_lines:]])
    if stderr:
        lines.extend(["--- stderr tail ---", *stderr.splitlines()[-max_lines:]])
    return "\n".join(lines) if lines else "(command produced no captured output)"


pipeline = mdr.read_files(RAW_INPUT).map(estimate_megasam).write_jsonl(HF_OUTPUT)


if __name__ == "__main__":
    pipeline.launch_cloud(
        name=os.environ.get("REFINER_JOB_NAME", "megasam-egocentric-inline"),
        num_workers=int(os.environ.get("REFINER_NUM_WORKERS", "1")),
        cpus_per_worker=int(os.environ.get("REFINER_CPUS_PER_WORKER", "8")),
        mem_mb_per_worker=int(os.environ.get("REFINER_MEM_MB_PER_WORKER", "65536")),
        gpu=mdr.GPU(count=1, type="h100", cuda_version="12.4"),
        secrets=mdr.Secrets.env(name="default", keys=["HF_TOKEN"]),
        env={
            "MACRODATA_BASE_URL": "https://dev.macrodata.co",
            "REFINER_IMG_FOCAL": IMG_FOCAL or "",
            "REFINER_EGO_ARTIFACT_REPO": HF_ARTIFACT_REPO or "",
            "REFINER_EGO_ARTIFACT_PATH": HF_ARTIFACT_PATH,
            "REFINER_METRIC_DEPTH_BACKEND": METRIC_DEPTH_BACKEND,
            "REFINER_LINGBOT_DEPTH_MODEL": LINGBOT_MODEL,
        },
    )
