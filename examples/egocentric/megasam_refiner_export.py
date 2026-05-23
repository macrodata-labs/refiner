from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--work-dir", default="/tmp/refiner-megasam-work")
    parser.add_argument("--scene-name", default="refiner_scene")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--focal", type=float)
    parser.add_argument("--skip-depth", action="store_true")
    parser.add_argument(
        "--metric-depth-backend",
        choices=("unidepth", "lingbot"),
        default="unidepth",
    )
    parser.add_argument(
        "--lingbot-model",
        default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5",
    )
    parser.add_argument("--weights", default="checkpoints/megasam_final.pth")
    parser.add_argument(
        "--depth-anything-weights",
        default="Depth-Anything/checkpoints/depth_anything_vitl14.pth",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    os.chdir(root)

    work_dir = Path(args.work_dir)
    scene = args.scene_name
    frame_dir = work_dir / "frames" / scene
    mono_depth_root = work_dir / "Depth-Anything" / "video_visualization"
    metric_depth_root = work_dir / "UniDepth" / "outputs"
    raw_metric_depth_root = work_dir / "UniDepth" / "raw_outputs"
    _reset_dir(frame_dir)
    mono_depth_root.mkdir(parents=True, exist_ok=True)
    metric_depth_root.mkdir(parents=True, exist_ok=True)
    raw_metric_depth_root.mkdir(parents=True, exist_ok=True)

    timestamps = _extract_frames(
        Path(args.video),
        frame_dir,
        max_frames=args.max_frames,
        stride=args.stride,
    )
    if not timestamps:
        raise RuntimeError("no frames extracted for MegaSAM")
    print(
        f"[refiner-megasam] extracted {len(timestamps)} frames to {frame_dir}",
        flush=True,
    )

    if not args.skip_depth:
        print("[refiner-megasam] running Depth-Anything", flush=True)
        _run(
            [
                sys.executable,
                "Depth-Anything/run_videos.py",
                "--encoder",
                "vitl",
                "--load-from",
                args.depth_anything_weights,
                "--img-path",
                str(frame_dir),
                "--outdir",
                str(mono_depth_root / scene),
            ]
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{root / 'UniDepth'}"
        if args.metric_depth_backend == "unidepth":
            print("[refiner-megasam] running UniDepth metric depth", flush=True)
            _run(
                [
                    sys.executable,
                    "UniDepth/scripts/demo_mega-sam.py",
                    "--scene-name",
                    scene,
                    "--img-path",
                    str(frame_dir),
                    "--outdir",
                    str(metric_depth_root),
                ],
                env=env,
            )
        else:
            print("[refiner-megasam] running UniDepth raw metric depth", flush=True)
            _run(
                [
                    sys.executable,
                    "UniDepth/scripts/demo_mega-sam.py",
                    "--scene-name",
                    scene,
                    "--img-path",
                    str(frame_dir),
                    "--outdir",
                    str(raw_metric_depth_root),
                ],
                env=env,
            )
            print("[refiner-megasam] running LingBot-Depth metric depth", flush=True)
            _run_lingbot_metric_depth(
                frame_dir=frame_dir,
                raw_depth_root=raw_metric_depth_root / scene,
                output_root=metric_depth_root,
                scene=scene,
                model_name=args.lingbot_model,
                focal=args.focal,
            )

    print("[refiner-megasam] running MegaSAM camera tracking", flush=True)
    _run(
        [
            sys.executable,
            "camera_tracking_scripts/test_demo.py",
            "--datapath",
            str(frame_dir),
            "--weights",
            args.weights,
            "--scene_name",
            scene,
            "--mono_depth_path",
            str(mono_depth_root),
            "--metric_depth_path",
            str(metric_depth_root),
            "--disable_vis",
        ]
    )

    output_npz = root / "outputs" / f"{scene}_droid.npz"
    payload = _payload_from_npz(
        output_npz,
        timestamps=timestamps,
        depth_backend=args.metric_depth_backend,
        metric_depth_root=metric_depth_root / scene,
        mono_depth_root=mono_depth_root / scene,
    )
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[refiner-megasam] wrote {result_path}", flush=True)


def _extract_frames(
    video_path: Path,
    output_dir: Path,
    *,
    max_frames: int,
    stride: int,
) -> list[float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    timestamps: list[float] = []
    frame_index = 0
    kept_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            path = output_dir / f"{kept_index:06d}.jpg"
            cv2.imwrite(str(path), frame)
            timestamps.append(frame_index / float(fps))
            kept_index += 1
            if max_frames > 0 and kept_index >= max_frames:
                break
        frame_index += 1
    capture.release()
    return timestamps


def _payload_from_npz(
    path: Path,
    *,
    timestamps: list[float],
    depth_backend: str,
    metric_depth_root: Path,
    mono_depth_root: Path,
) -> dict:
    data = np.load(path, allow_pickle=False)
    cam_c2w = np.asarray(data["cam_c2w"], dtype=np.float64)
    if len(cam_c2w) != len(timestamps):
        timestamps = timestamps[: len(cam_c2w)]
    payload: dict = {
        "T_world_camera": cam_c2w.tolist(),
        "source": "megasam",
        "timestamps": timestamps,
    }
    if "intrinsic" in data:
        payload["intrinsic"] = np.asarray(data["intrinsic"], dtype=np.float64).tolist()
    if "depths" in data:
        depths = np.asarray(data["depths"], dtype=np.float32)
        payload["depth"] = {
            "source": f"{depth_backend}+depth_anything",
            "timestamps": timestamps,
            "metric_depth": {
                "format": "megasam_metric_depth_npz",
                "frame_count": int(len(timestamps)),
                "directory": str(metric_depth_root),
                "backend": depth_backend,
            },
            "mono_depth": {
                "format": "depth_anything_npy",
                "frame_count": int(len(timestamps)),
                "directory": str(mono_depth_root),
            },
            "tracked_depth": {
                "format": "megasam_output_npz_depths",
                "shape": list(depths.shape),
                "min": float(np.nanmin(depths)),
                "max": float(np.nanmax(depths)),
                "median": float(np.nanmedian(depths)),
            },
        }
    return payload


def _run_lingbot_metric_depth(
    *,
    frame_dir: Path,
    raw_depth_root: Path,
    output_root: Path,
    scene: str,
    model_name: str,
    focal: float | None,
) -> None:
    import torch
    from PIL import Image
    from mdm.model.v2 import MDMModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDMModel.from_pretrained(model_name).to(device)
    model.eval()
    output_dir = output_root / scene
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
    with torch.inference_mode():
        for image_path in image_paths:
            image = np.array(Image.open(image_path).convert("RGB"))
            raw_depth_path = raw_depth_root / f"{image_path.stem}.npz"
            if not raw_depth_path.exists():
                raise FileNotFoundError(
                    f"missing raw depth for LingBot: {raw_depth_path}"
                )
            raw_depth = np.asarray(np.load(raw_depth_path)["depth"], dtype=np.float32)
            h, w = image.shape[:2]
            frame_focal = float(focal) if focal is not None else float(max(w, h))
            intrinsics = np.array(
                [
                    [frame_focal / w, 0.0, 0.5],
                    [0.0, frame_focal / h, 0.5],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            image_t = (
                torch.tensor(image / 255.0, dtype=torch.float32, device=device)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            depth_t = torch.tensor(raw_depth, dtype=torch.float32, device=device)[None]
            intrinsics_t = torch.tensor(intrinsics, dtype=torch.float32, device=device)[
                None
            ]
            output = model.infer(
                image_t,
                depth_in=depth_t,
                intrinsics=intrinsics_t,
                use_fp16=torch.cuda.is_available(),
            )
            depth = output["depth"]
            if depth.ndim == 4:
                depth = depth[0, 0]
            elif depth.ndim == 3:
                depth = depth[0]
            depth_np = depth.detach().float().cpu().numpy().astype(np.float32)
            fov = float(np.rad2deg(2.0 * np.arctan(w / (2.0 * frame_focal))))
            np.savez(output_dir / f"{image_path.stem}.npz", depth=depth_np, fov=fov)


def _run(args: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"[refiner-megasam] command: {' '.join(args)}", flush=True)
    completed = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed: "
            + " ".join(args)
            + f"\nexit code: {completed.returncode}"
            + "\n"
            + _tail(completed.stdout, completed.stderr)
        )


def _tail(stdout: str | None, stderr: str | None, *, max_lines: int = 120) -> str:
    lines = []
    if stdout:
        lines.extend(["--- stdout tail ---", *stdout.splitlines()[-max_lines:]])
    if stderr:
        lines.extend(["--- stderr tail ---", *stderr.splitlines()[-max_lines:]])
    return "\n".join(lines)


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
