from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VGGT-Omega and write a Refiner geometry artifact."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--result", required=True, help="Output Refiner JSON path.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for intermediate frames and raw VGGT-Omega npz output.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a VGGT-Omega checkpoint, for example model.pt.",
    )
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Optional local facebookresearch/vggt-omega checkout to import from.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--text-alignment",
        action="store_true",
        help="Use VGGTOmega(enable_alignment=True); intended for the 256 text model.",
    )
    args = parser.parse_args()

    if args.repo_path:
        sys.path.insert(0, str(Path(args.repo_path).resolve()))

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        frame_dir = Path(tmp) / "frames"
        frame_dir.mkdir()
        image_paths, timestamps = _extract_video_frames(
            video_path=Path(args.video),
            frame_dir=frame_dir,
            stride=args.stride,
            max_frames=args.max_frames,
        )

        if not image_paths:
            raise RuntimeError("no frames were extracted from the input video")

        device = torch.device(args.device)
        model = VGGTOmega(enable_alignment=args.text_alignment).to(device).eval()
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)

        images = load_and_preprocess_images(
            [str(path) for path in image_paths],
            image_resolution=args.image_resolution,
        ).to(device)

        with torch.inference_mode():
            predictions = model(images)

        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )

        raw_npz = output_dir / "vggt_omega_raw.npz"
        np.savez_compressed(
            raw_npz,
            extrinsics=_to_numpy(extrinsics),
            intrinsics=_to_numpy(intrinsics),
            depth=_to_numpy(predictions["depth"]),
            depth_conf=_to_numpy(predictions["depth_conf"]),
            timestamps=np.asarray(timestamps, dtype=np.float64),
            image_size=np.asarray(predictions["images"].shape[-2:], dtype=np.int64),
        )

    result = _write_vggt_omega_geometry_json(
        npz_path=raw_npz,
        output_path=args.result,
        extrinsic_convention="world_to_camera",
    )
    print(json.dumps({"result": str(result), "raw_npz": str(raw_npz)}))


def _extract_video_frames(
    *,
    video_path: Path,
    frame_dir: Path,
    stride: int,
    max_frames: int | None,
) -> tuple[list[Path], list[float]]:
    if stride <= 0:
        raise ValueError("--stride must be > 0")
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "vggt_omega_refiner_export.py requires PyAV to decode videos. "
            "Install Refiner with the video extra or install av."
        ) from exc

    image_paths: list[Path] = []
    timestamps: list[float] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index % stride != 0:
                continue
            image = frame.to_image()
            path = frame_dir / f"{len(image_paths):06d}.jpg"
            image.save(path, quality=95)
            image_paths.append(path)
            if frame.pts is not None and frame.time_base is not None:
                timestamps.append(float(frame.pts * frame.time_base))
            else:
                timestamps.append(float(frame_index))
            if max_frames is not None and len(image_paths) >= max_frames:
                break
    return image_paths, timestamps


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
    array = np.asarray(value)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    return array


def _write_vggt_omega_geometry_json(
    *,
    npz_path: Path,
    output_path: str,
    extrinsic_convention: str,
) -> Path:
    payload = _geometry_payload_from_vggt_omega_npz(
        npz_path,
        extrinsic_convention=extrinsic_convention,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def _geometry_payload_from_vggt_omega_npz(
    path: Path,
    *,
    extrinsic_convention: str,
) -> dict:
    data = np.load(path, allow_pickle=False)
    transforms = _load_transforms(data, extrinsic_convention=extrinsic_convention)
    frame_count = int(transforms.shape[0])
    timestamps = _load_timestamps(data, frame_count=frame_count)

    camera = {
        "source": "vggt-omega",
        "T_world_camera": transforms.tolist(),
    }
    intrinsics = _optional_array(data, ("intrinsics", "intrinsic", "K"))
    if intrinsics is not None:
        if intrinsics.shape == (3, 3):
            camera["intrinsic"] = intrinsics.astype(np.float64).tolist()
        elif intrinsics.shape == (frame_count, 3, 3):
            camera["intrinsics"] = intrinsics.astype(np.float64).tolist()
        else:
            raise ValueError("VGGT-Omega intrinsics must be shaped 3x3 or Tx3x3")
    image_size = _optional_array(data, ("image_size", "image_shape"))
    if image_size is not None:
        camera["image_size"] = np.asarray(image_size).astype(int).tolist()

    depth = {
        "source": "vggt-omega",
        "timestamps": timestamps,
        "metric_depth": {
            "format": "vggt_omega_npz",
            "frame_count": frame_count,
            "artifact": str(path),
        },
    }
    depth_key = _first_present_key(data, ("depth", "depths", "depth_map", "depth_maps"))
    if depth_key is not None:
        depth["metric_depth"]["key"] = depth_key
    confidence_key = _first_present_key(
        data,
        ("depth_conf", "depth_confidence", "depth_confs"),
    )
    if confidence_key is not None:
        depth["metric_depth"]["confidence_key"] = confidence_key

    return {
        "source": "vggt-omega",
        "timestamps": timestamps,
        "camera": camera,
        "depth": depth,
        "artifact": str(path),
    }


def _load_transforms(data, *, extrinsic_convention: str) -> np.ndarray:
    key = _first_present_key(data, ("extrinsics", "extrinsic", "poses", "camera_poses"))
    if key is None:
        raise ValueError("VGGT-Omega npz requires extrinsics or poses")
    transforms = np.asarray(data[key], dtype=np.float64)
    if transforms.ndim == 2 and transforms.shape == (4, 4):
        transforms = transforms[None, :, :]
    if transforms.ndim == 2 and transforms.shape == (3, 4):
        transforms = transforms[None, :, :]
    if transforms.ndim == 3 and transforms.shape[1:] == (3, 4):
        homogeneous = np.tile(
            np.eye(4, dtype=np.float64)[None, :, :],
            (transforms.shape[0], 1, 1),
        )
        homogeneous[:, :3, :] = transforms
        transforms = homogeneous
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("VGGT-Omega extrinsics must be shaped Tx4x4")
    if extrinsic_convention == "world_to_camera":
        transforms = np.linalg.inv(transforms)
    elif extrinsic_convention != "camera_to_world":
        raise ValueError(
            "extrinsic_convention must be world_to_camera or camera_to_world"
        )
    return transforms


def _load_timestamps(data, *, frame_count: int) -> list[float]:
    timestamps = _optional_array(data, ("timestamps", "timestamp", "times"))
    if timestamps is None:
        return [float(i) for i in range(frame_count)]
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if len(timestamps) != frame_count:
        raise ValueError("VGGT-Omega timestamps length must match frame count")
    return timestamps.tolist()


def _optional_array(data, keys: tuple[str, ...]) -> np.ndarray | None:
    key = _first_present_key(data, keys)
    if key is None:
        return None
    return np.asarray(data[key])


def _first_present_key(data, keys: tuple[str, ...]) -> str | None:
    available = set(data.files)
    for key in keys:
        if key in available:
            return key
    return None


if __name__ == "__main__":
    main()
