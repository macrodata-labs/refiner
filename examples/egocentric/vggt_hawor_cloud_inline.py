from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import refiner as mdr


RAW_INPUT = os.environ["REFINER_EGO_INPUT"]
HF_OUTPUT = os.environ["REFINER_EGO_OUTPUT"]
HF_ARTIFACT_REPO = os.environ.get(
    "REFINER_EGO_ARTIFACT_REPO",
    "macrodata/egocentric-vggt-debug",
)
HF_ARTIFACT_PREFIX = os.environ.get("REFINER_EGO_ARTIFACT_PREFIX", "egodex-vggt")
CLIP_SECONDS = float(os.environ.get("REFINER_EGO_CLIP_SECONDS", "10"))
IMG_FOCAL = os.environ.get("REFINER_IMG_FOCAL")

HAWOR_EXPORT = os.environ.get("REFINER_HAWOR_EXPORT", "/opt/HaWoR/refiner_export.py")
HAWOR_SETUP = os.environ.get("REFINER_HAWOR_SETUP")
VGGT_OMEGA_EXPORT = os.environ.get(
    "REFINER_VGGT_OMEGA_EXPORT",
    "/tmp/refiner-vggt-omega-runtime/vggt_omega_refiner_export.py",
)
VGGT_OMEGA_SETUP = os.environ.get("REFINER_VGGT_OMEGA_SETUP")
VGGT_OMEGA_REPO = os.environ.get("REFINER_VGGT_OMEGA_REPO", "/opt/vggt-omega")
VGGT_OMEGA_CHECKPOINT = os.environ.get("REFINER_VGGT_OMEGA_CHECKPOINT")
VGGT_OMEGA_RESOLUTION = int(os.environ.get("REFINER_VGGT_OMEGA_RESOLUTION", "512"))
VGGT_OMEGA_MAX_FRAMES = int(os.environ.get("REFINER_VGGT_OMEGA_MAX_FRAMES", "0"))
VGGT_OMEGA_STRIDE = int(os.environ.get("REFINER_VGGT_OMEGA_STRIDE", "1"))

_SCRIPT_DIR = Path(__file__).resolve().parent
_HAWOR_READY = False
_VGGT_READY = False
_BUNDLED_HAWOR_RUNTIME_FILES = {
    "install_hawor_official_runtime.sh": (
        _SCRIPT_DIR / "install_hawor_official_runtime.sh"
    ).read_text(),
    "hawor_refiner_export.py": (_SCRIPT_DIR / "hawor_refiner_export.py").read_text(),
}
_BUNDLED_VGGT_RUNTIME_FILES = {
    "install_vggt_omega_runtime.sh": (
        _SCRIPT_DIR / "install_vggt_omega_runtime.sh"
    ).read_text(),
    "vggt_omega_refiner_export.py": (
        _SCRIPT_DIR / "vggt_omega_refiner_export.py"
    ).read_text(),
}


def run_vggt_hawor(row):
    ensure_vggt_runtime()

    video_path = _materialize_video_path(str(row["file_path"]))
    output_dir = Path("/tmp/vggt-hawor-artifacts") / "sample"
    output_dir.mkdir(parents=True, exist_ok=True)
    if CLIP_SECONDS > 0:
        clip_path = output_dir / "input_clip.mp4"
        _write_clip(video_path, clip_path, CLIP_SECONDS)
        video_path = str(clip_path)

    checkpoint = VGGT_OMEGA_CHECKPOINT or _download_vggt_checkpoint()
    ensure_hawor_runtime()

    vggt_result = output_dir / "vggt_omega_geometry.json"
    vggt_args = [
        "python",
        VGGT_OMEGA_EXPORT,
        "--video",
        video_path,
        "--output-dir",
        str(output_dir / "vggt"),
        "--result",
        str(vggt_result),
        "--checkpoint",
        checkpoint,
        "--repo-path",
        VGGT_OMEGA_REPO,
        "--image-resolution",
        str(VGGT_OMEGA_RESOLUTION),
    ]
    if VGGT_OMEGA_MAX_FRAMES > 0:
        vggt_args.extend(["--max-frames", str(VGGT_OMEGA_MAX_FRAMES)])
    if VGGT_OMEGA_STRIDE != 1:
        vggt_args.extend(["--stride", str(VGGT_OMEGA_STRIDE)])
    _run_command(vggt_args, name="VGGT-Omega export")
    vggt_payload = json.loads(vggt_result.read_text())
    vggt_img_focal = _focal_from_vggt_camera(vggt_payload["camera"])

    hawor_result = output_dir / "hawor_camera_hands.json"
    hawor_args = [
        "python",
        HAWOR_EXPORT,
        "--video",
        video_path,
        "--result",
        str(hawor_result),
    ]
    img_focal = float(IMG_FOCAL) if IMG_FOCAL else vggt_img_focal
    if img_focal:
        hawor_args.extend(["--img_focal", str(img_focal)])
    _run_command(
        hawor_args,
        name="HaWoR export",
        env={**os.environ, "HAWOR_SKIP_RENDERER": "1"},
    )

    fused_path = output_dir / "vggt_hawor_fused.json"
    fused = _fuse_and_export_rerun(
        vggt_path=vggt_result,
        hawor_path=hawor_result,
        video_path=video_path,
        fused_path=fused_path,
    )

    urls = _upload_artifacts(
        {
            f"{HF_ARTIFACT_PREFIX}/input_clip.mp4": Path(video_path),
            f"{HF_ARTIFACT_PREFIX}/vggt_omega_geometry.json": vggt_result,
            f"{HF_ARTIFACT_PREFIX}/hawor_camera_hands.json": hawor_result,
            f"{HF_ARTIFACT_PREFIX}/vggt_hawor_fused.json": fused_path,
        }
    )
    return row.update(
        {
            "vggt_omega": json.loads(vggt_result.read_text()),
            "hawor": json.loads(hawor_result.read_text()),
            "vggt_hawor": fused,
            "artifact_urls": urls,
        }
    )


def ensure_hawor_runtime() -> None:
    global _HAWOR_READY
    if _HAWOR_READY:
        return
    if not Path(HAWOR_EXPORT).exists():
        runtime_dir = _write_runtime(
            "/tmp/refiner-hawor-official-runtime",
            _BUNDLED_HAWOR_RUNTIME_FILES,
        )
        setup = HAWOR_SETUP
        if not setup:
            setup = (
                "HAWOR_FORCE_TORCH_INSTALL=1 "
                "HAWOR_SKIP_RENDERER=1 "
                "HAWOR_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 "
                f"REFINER_EXPORT={runtime_dir / 'hawor_refiner_export.py'} "
                f"bash {runtime_dir / 'install_hawor_official_runtime.sh'}"
            )
        _run_command(setup, name="HaWoR setup", shell=True)
    _HAWOR_READY = True


def ensure_vggt_runtime() -> None:
    global _VGGT_READY
    if _VGGT_READY:
        return
    runtime_dir = _write_runtime(
        "/tmp/refiner-vggt-omega-runtime",
        _BUNDLED_VGGT_RUNTIME_FILES,
    )
    if not Path(VGGT_OMEGA_REPO).exists():
        setup = VGGT_OMEGA_SETUP
        if not setup:
            setup = (
                f"VGGT_OMEGA_REPO={VGGT_OMEGA_REPO} "
                f"bash {runtime_dir / 'install_vggt_omega_runtime.sh'}"
            )
        _run_command(setup, name="VGGT-Omega setup", shell=True)
    _VGGT_READY = True


def _write_runtime(root: str, files: dict[str, str]) -> Path:
    runtime_dir = Path(root)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for name, contents in files.items():
        path = runtime_dir / name
        path.write_text(contents)
        if name.endswith(".sh"):
            path.chmod(0o755)
    return runtime_dir


def _download_vggt_checkpoint() -> str:
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    repo_id = "facebook/VGGT-Omega"
    api = HfApi(token=token)
    info = api.model_info(repo_id=repo_id, files_metadata=True)
    files = {s.rfilename: s.size for s in info.siblings}
    candidates = [
        "vggt_omega_1b_512.pt",
        "vggt_omega_1b_256_text.pt",
        "VGGT-Omega-1B-512/model.pt",
        "checkpoints/VGGT-Omega-1B-512/model.pt",
        "model.pt",
    ]
    for filename in candidates:
        if filename in files:
            cli_path = _download_vggt_checkpoint_with_hf_cli(
                repo_id=repo_id,
                filename=filename,
                token=token,
                disable_xet=False,
            )
            if cli_path:
                return cli_path
            cli_path = _download_vggt_checkpoint_with_hf_cli(
                repo_id=repo_id,
                filename=filename,
                token=token,
                disable_xet=True,
            )
            if cli_path:
                return cli_path
            last_error = None
            for attempt in range(1, 8):
                try:
                    return hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        token=token,
                        etag_timeout=60,
                        resume_download=True,
                    )
                except Exception as exc:
                    last_error = exc
                    print(
                        "[refiner-vggt-hawor] checkpoint download failed "
                        f"attempt={attempt}: {exc}",
                        flush=True,
                    )
                    time.sleep(10 * attempt)
            raise RuntimeError(
                f"Could not download VGGT-Omega checkpoint {filename}: {last_error}"
            ) from last_error
    raise RuntimeError(
        "Could not find VGGT-Omega checkpoint in facebook/VGGT-Omega. "
        f"Available files include: {sorted(files)[:20]}"
    )


def _download_vggt_checkpoint_with_hf_cli(
    *,
    repo_id: str,
    filename: str,
    token: str | None,
    disable_xet: bool,
) -> str | None:
    local_dir = Path("/tmp/vggt-omega-checkpoints")
    local_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
    if disable_xet:
        env["HF_HUB_DISABLE_XET"] = "1"
    cmd = [
        "hf",
        "download",
        repo_id,
        filename,
        "--repo-type",
        "model",
        "--local-dir",
        str(local_dir),
        "--max-workers",
        "4",
    ]
    if token:
        cmd.extend(["--token", token])
    try:
        _run_command(
            cmd,
            name=(
                "VGGT-Omega checkpoint download"
                + (" without Xet" if disable_xet else "")
            ),
            env=env,
        )
    except Exception as exc:
        print(
            "[refiner-vggt-hawor] hf CLI checkpoint download failed "
            f"disable_xet={disable_xet}: {exc}",
            flush=True,
        )
        return None
    path = local_dir / filename
    if path.exists():
        return str(path)
    return None


def _download_vggt_checkpoint_direct(
    *,
    url: str,
    filename: str,
    token: str | None,
    expected_size: int | None,
) -> str | None:
    import requests

    local_dir = Path("/tmp/vggt-omega-checkpoints-direct")
    local_dir.mkdir(parents=True, exist_ok=True)
    path = local_dir / filename
    last_error: Exception | None = None
    for attempt in range(1, 11):
        downloaded = path.stat().st_size if path.exists() else 0
        if expected_size and downloaded >= expected_size:
            return str(path)
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if downloaded:
            headers["Range"] = f"bytes={downloaded}-"
        print(
            "[refiner-vggt-hawor] direct checkpoint download "
            f"attempt={attempt} downloaded={downloaded} expected={expected_size}",
            flush=True,
        )
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 600),
            ) as response:
                if downloaded and response.status_code == 200:
                    downloaded = 0
                response.raise_for_status()
                mode = "ab" if downloaded and response.status_code == 206 else "wb"
                with path.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            size = path.stat().st_size
            if not expected_size or size >= expected_size:
                return str(path)
            last_error = RuntimeError(
                f"incomplete checkpoint download: {size} of {expected_size} bytes"
            )
        except Exception as exc:
            last_error = exc
            print(
                "[refiner-vggt-hawor] direct checkpoint download failed "
                f"attempt={attempt}: {exc}",
                flush=True,
            )
        time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"Direct VGGT-Omega checkpoint download failed: {last_error}")


def _fuse_and_export_rerun(
    *,
    vggt_path: Path,
    hawor_path: Path,
    video_path: str,
    fused_path: Path,
) -> dict:
    vggt = json.loads(vggt_path.read_text())
    result = json.loads(hawor_path.read_text())
    t_world_camera = vggt["camera"]["T_world_camera"]
    for side in ("left", "right"):
        hand = result.get(f"{side}_hand")
        if isinstance(hand, dict):
            _project_hand_to_world(hand, t_world_camera)
    result["camera"] = vggt["camera"]
    result["depth"] = vggt["depth"]
    vggt_img_focal = _focal_from_vggt_camera(vggt["camera"])
    result["metadata"] = {
        **(result.get("metadata") or {}),
        "geometry_provider": "vggt-omega",
        "video_path": video_path,
        "img_focal": vggt_img_focal or (result.get("metadata") or {}).get("img_focal"),
        "vggt_img_focal": vggt_img_focal,
        "hawor_img_focal": (result.get("metadata") or {}).get("img_focal"),
    }
    fused_path.write_text(json.dumps(result), encoding="utf-8")
    return result


def _project_hand_to_world(hand: dict, t_world_camera: list) -> None:
    import numpy as np

    t_world_camera_np = np.asarray(t_world_camera, dtype=np.float64)
    if t_world_camera_np.ndim != 3 or t_world_camera_np.shape[1:] != (4, 4):
        raise ValueError("VGGT camera.T_world_camera must be shaped Tx4x4")
    frame_count = t_world_camera_np.shape[0]
    if "T_camera_wrist" in hand:
        t_camera_wrist = np.asarray(hand["T_camera_wrist"], dtype=np.float64)
        count = min(frame_count, len(t_camera_wrist))
        projected = np.matmul(t_world_camera_np[:count], t_camera_wrist[:count])
        hand["T_world_wrist"] = projected.tolist()
    if "joints_camera" in hand:
        joints_camera = np.asarray(hand["joints_camera"], dtype=np.float64)
        count = min(frame_count, len(joints_camera))
        ones = np.ones((*joints_camera[:count].shape[:2], 1), dtype=np.float64)
        homogeneous = np.concatenate([joints_camera[:count], ones], axis=-1)
        projected = np.einsum(
            "tij,tkj->tki",
            t_world_camera_np[:count],
            homogeneous,
        )[..., :3]
        hand["joints_world"] = projected.tolist()


def _focal_from_vggt_camera(camera: dict) -> float | None:
    intrinsic = camera.get("intrinsic")
    if intrinsic:
        return float((float(intrinsic[0][0]) + float(intrinsic[1][1])) / 2.0)
    intrinsics = camera.get("intrinsics")
    if intrinsics:
        first = intrinsics[0]
        return float((float(first[0][0]) + float(first[1][1])) / 2.0)
    return None


def _upload_artifacts(paths: dict[str, Path]) -> dict[str, str]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(
        repo_id=HF_ARTIFACT_REPO,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    urls = {}
    for path_in_repo, local_path in paths.items():
        urls[path_in_repo] = api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=HF_ARTIFACT_REPO,
            repo_type="dataset",
        )
    return urls


def _hawor_renderer_available() -> bool:
    hawor_root = Path(HAWOR_EXPORT).resolve().parent
    if not hawor_root.exists():
        return False
    completed = subprocess.run(
        ["python", "-c", "import pytorch3d; from lib.vis.renderer import Renderer"],
        cwd=hawor_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _materialize_video_path(video_path: str) -> str:
    if not video_path.startswith("hf://datasets/"):
        return video_path
    from huggingface_hub import hf_hub_download

    rest = video_path.removeprefix("hf://datasets/")
    namespace, name, filename = rest.split("/", 2)
    return hf_hub_download(
        repo_id=f"{namespace}/{name}",
        repo_type="dataset",
        filename=filename,
        token=os.environ.get("HF_TOKEN"),
    )


def _write_clip(input_path: str, output_path: Path, seconds: float) -> None:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    _run_command(
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
        name="video clipping",
    )
    if output_path.stat().st_size == 0:
        raise RuntimeError("video clipping failed: empty output")


def _run_command(args, *, name: str, shell: bool = False, env=None) -> None:
    print(f"[refiner-vggt-hawor] running {name}: {args}", flush=True)
    completed = subprocess.run(
        args,
        shell=shell,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {completed.returncode}\n"
            f"{_tail_output(completed.stdout, completed.stderr)}"
        )


def _tail_output(
    stdout: str | None, stderr: str | None, *, max_lines: int = 160
) -> str:
    lines = []
    if stdout:
        lines.extend(["--- stdout tail ---", *stdout.splitlines()[-max_lines:]])
    if stderr:
        lines.extend(["--- stderr tail ---", *stderr.splitlines()[-max_lines:]])
    return "\n".join(lines) if lines else "(command produced no captured output)"


pipeline = mdr.read_files(RAW_INPUT).map(run_vggt_hawor).write_jsonl(HF_OUTPUT)


if __name__ == "__main__":
    secret_sources = [mdr.Secrets.env(name="default", keys=["HF_TOKEN"])]
    local_hf_token = os.environ.get("REFINER_HF_TOKEN") or os.environ.get("HF_TOKEN")
    if local_hf_token:
        secret_sources.append({"HF_TOKEN": local_hf_token})
    pipeline.launch_cloud(
        name=os.environ.get("REFINER_JOB_NAME", "vggt-omega-hawor-egocentric"),
        num_workers=int(os.environ.get("REFINER_NUM_WORKERS", "1")),
        cpus_per_worker=int(os.environ.get("REFINER_CPUS_PER_WORKER", "8")),
        mem_mb_per_worker=int(os.environ.get("REFINER_MEM_MB_PER_WORKER", "65536")),
        gpu=mdr.GPU(count=1, type="h100", cuda_version="12.4"),
        secrets=secret_sources,
        env={
            "MACRODATA_BASE_URL": "https://dev.macrodata.co",
            "REFINER_EGO_ARTIFACT_REPO": HF_ARTIFACT_REPO,
            "REFINER_EGO_ARTIFACT_PREFIX": HF_ARTIFACT_PREFIX,
            "REFINER_EGO_CLIP_SECONDS": str(CLIP_SECONDS),
            "REFINER_IMG_FOCAL": IMG_FOCAL or "",
            "REFINER_VGGT_OMEGA_RESOLUTION": str(VGGT_OMEGA_RESOLUTION),
            "REFINER_VGGT_OMEGA_MAX_FRAMES": str(VGGT_OMEGA_MAX_FRAMES),
            "REFINER_VGGT_OMEGA_STRIDE": str(VGGT_OMEGA_STRIDE),
            "REFINER_VGGT_OMEGA_CHECKPOINT": VGGT_OMEGA_CHECKPOINT or "",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_DOWNLOAD_TIMEOUT": "600",
        },
    )
