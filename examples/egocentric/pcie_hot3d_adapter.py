from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_STREAM_ID = "214-1"
DEFAULT_PCIE_REPO = Path("/private/tmp/PCIE_EgoHandPose")


def prepare_hot3d_as_pcie_test(args: argparse.Namespace) -> None:
    tar_paths = _resolve_tar_paths(args.hot3d_root, args.clips, args.limit)
    anno_dir = args.output_root / "annotation"
    image_root = args.output_root / "image" / "undistorted"
    test_image_root = image_root / "test"
    anno_dir.mkdir(parents=True, exist_ok=True)
    test_image_root.mkdir(parents=True, exist_ok=True)

    annotations: dict[str, dict[str, Any]] = {}
    manifest: list[dict[str, Any]] = []
    for tar_path in tar_paths:
        take_uid = tar_path.stem
        take_name = take_uid
        take_image_dir = test_image_root / take_name
        take_image_dir.mkdir(parents=True, exist_ok=True)
        take_anno, rows = _convert_tar(
            tar_path,
            take_uid=take_uid,
            take_name=take_name,
            image_dir=take_image_dir,
            stream_id=args.stream_id,
            bbox_format=args.bbox_format,
            min_visibility=args.min_visibility,
            max_frames=args.max_frames_per_clip,
            overwrite=args.overwrite,
        )
        if take_anno:
            annotations[take_uid] = take_anno
            manifest.extend(rows)

    anno_path = anno_dir / "ego_pose_gt_anno_test_public.json"
    anno_path.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    manifest_path = args.output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row) + "\n")

    print(f"wrote annotation: {anno_path}")
    print(f"wrote images: {test_image_root}")
    print(f"wrote manifest: {manifest_path}")
    print(f"frames with at least one bbox: {sum(len(v) for v in annotations.values())}")
    print("PCIE arguments:")
    print(f"  --gt_anno_dir {anno_dir}")
    print(f"  --aria_img_dir {image_root}")


def prepare_hot3d_detector_tracks_as_pcie_test(args: argparse.Namespace) -> None:
    """Prepare PCIE test inputs from saved HaWoR detector/tracker boxes.

    This mode intentionally does not read HOT3D ``hands.json`` or
    ``hand_crops.json``. Boxes come from HaWoR's saved ``model_tracks.npy``
    detector/tracker output, which is model-derived and suitable for the clean
    PCIE side track.
    """

    tar_paths = _resolve_tar_paths(args.hot3d_root, args.clips, args.limit)
    anno_dir = args.output_root / "annotation"
    image_root = args.output_root / "image" / "undistorted"
    test_image_root = image_root / "test"
    anno_dir.mkdir(parents=True, exist_ok=True)
    test_image_root.mkdir(parents=True, exist_ok=True)

    annotations: dict[str, dict[str, Any]] = {}
    manifest: list[dict[str, Any]] = []
    for tar_path in tar_paths:
        take_uid = tar_path.stem
        take_name = take_uid
        tracks_path = (
            args.tracks_root
            / take_uid
            / f"rgb_{args.stream_id}"
            / args.tracks_subdir
            / "model_tracks.npy"
        )
        if not tracks_path.exists():
            raise FileNotFoundError(f"missing detector tracks: {tracks_path}")
        take_image_dir = test_image_root / take_name
        take_image_dir.mkdir(parents=True, exist_ok=True)
        take_anno, rows = _convert_tar_with_detector_tracks(
            tar_path,
            tracks_path=tracks_path,
            take_uid=take_uid,
            take_name=take_name,
            image_dir=take_image_dir,
            stream_id=args.stream_id,
            max_frames=args.max_frames_per_clip,
            overwrite=args.overwrite,
            min_score=args.min_score,
            expand=args.expand,
        )
        if take_anno:
            annotations[take_uid] = take_anno
            manifest.extend(rows)

    anno_path = anno_dir / "ego_pose_gt_anno_test_public.json"
    anno_path.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    manifest_path = args.output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row) + "\n")

    print(f"wrote annotation: {anno_path}")
    print(f"wrote images: {test_image_root}")
    print(f"wrote manifest: {manifest_path}")
    print(
        f"frames with at least one detector bbox: {sum(len(v) for v in annotations.values())}"
    )
    print("PCIE arguments:")
    print(f"  --gt_anno_dir {anno_dir}")
    print(f"  --aria_img_dir {image_root}")


def run_pcie_inference(args: argparse.Namespace) -> None:
    repo = args.pcie_repo
    inference_py = repo / "inference.py"
    if not inference_py.exists():
        raise SystemExit(f"missing PCIE inference.py: {inference_py}")
    if not args.pretrained_ckpt.exists():
        raise SystemExit(f"missing PCIE checkpoint: {args.pretrained_ckpt}")
    if not args.cfg_file.exists():
        raise SystemExit(f"missing PCIE config: {args.cfg_file}")
    if not (
        args.prepared_root / "annotation" / "ego_pose_gt_anno_test_public.json"
    ).exists():
        raise SystemExit(
            "prepared annotations are missing; run the 'prepare' subcommand first"
        )

    command = [
        "python",
        str(inference_py),
        "--pretrained_ckpt",
        str(args.pretrained_ckpt),
        "--cfg_file",
        str(args.cfg_file),
        "--gt_anno_dir",
        str(args.prepared_root / "annotation"),
        "--aria_img_dir",
        str(args.prepared_root / "image" / "undistorted"),
        "--output_dir",
        str(args.output_dir),
    ]
    print("running:", " ".join(command))
    completed = subprocess.run(command, cwd=repo, check=False)
    raise SystemExit(completed.returncode)


def inspect_hot3d(args: argparse.Namespace) -> None:
    for tar_path in _resolve_tar_paths(args.hot3d_root, args.clips, args.limit):
        with tarfile.open(tar_path) as archive:
            names = archive.getnames()
            image_count = sum(
                name.endswith(f".image_{args.stream_id}.jpg") for name in names
            )
            hands_count = sum(name.endswith(".hands.json") for name in names)
            camera_count = sum(name.endswith(".cameras.json") for name in names)
            first_hands = next(
                (name for name in names if name.endswith(".hands.json")), None
            )
            hand_keys: dict[str, list[str]] = {}
            if first_hands:
                hands = _read_json(archive, first_hands)
                hand_keys = {
                    side: sorted(payload.keys())
                    for side, payload in hands.items()
                    if isinstance(payload, dict)
                }
            print(
                json.dumps(
                    {
                        "tar": str(tar_path),
                        "image_count": image_count,
                        "hands_json_count": hands_count,
                        "cameras_json_count": camera_count,
                        "sample_hand_keys": hand_keys,
                    },
                    indent=2,
                )
            )


def print_modal_commands(args: argparse.Namespace) -> None:
    prepared_root = args.prepared_root
    repo = args.pcie_repo
    ckpt = args.pretrained_ckpt
    cfg = args.cfg_file
    output_dir = args.output_dir
    print(
        "\n".join(
            [
                "# Create or reuse the existing HOT3D Modal sandbox.",
                "uv run python examples/egocentric/modal_hot3d_sandbox.py create --timeout-hours 4",
                "",
                "# In the Modal sandbox, install PCIE runtime and run a small clip set.",
                "git clone https://github.com/KanokphanL/PCIE_EgoHandPose.git /cache/PCIE_EgoHandPose || true",
                "cd /cache/PCIE_EgoHandPose",
                "python -m pip install -r requirement.txt",
                "python -m pip install mmdet==3.1.0 mmpretrain==1.2.0",
                "# Install a CUDA-matched torch/torchvision pair if the image does not already provide one.",
                "",
                "# Copy this adapter into the sandbox or run it from the mounted repo, then:",
                "python examples/egocentric/pcie_hot3d_adapter.py prepare \\",
                f"  --hot3d-root {args.hot3d_root} \\",
                f"  --output-root {prepared_root} \\",
                "  --clips clip-001849 \\",
                "  --max-frames-per-clip 30",
                "python examples/egocentric/pcie_hot3d_adapter.py run \\",
                f"  --pcie-repo {repo} \\",
                f"  --prepared-root {prepared_root} \\",
                f"  --pretrained-ckpt {ckpt} \\",
                f"  --cfg-file {cfg} \\",
                f"  --output-dir {output_dir}",
            ]
        )
    )


def _convert_tar(
    tar_path: Path,
    *,
    take_uid: str,
    take_name: str,
    image_dir: Path,
    stream_id: str,
    bbox_format: str,
    min_visibility: float,
    max_frames: int | None,
    overwrite: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    take_anno: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    with tarfile.open(tar_path) as archive:
        image_members = {
            _frame_id_from_name(name, f".image_{stream_id}.jpg"): name
            for name in archive.getnames()
            if name.endswith(f".image_{stream_id}.jpg")
        }
        hand_members = sorted(
            name for name in archive.getnames() if name.endswith(".hands.json")
        )
        if not image_members:
            raise ValueError(f"{tar_path} has no .image_{stream_id}.jpg entries")
        emitted = 0
        for sequence_index, hands_name in enumerate(hand_members):
            if max_frames is not None and emitted >= max_frames:
                break
            frame_id = _frame_id_from_name(hands_name, ".hands.json")
            image_name = image_members.get(frame_id)
            if image_name is None:
                continue
            hands = _read_json(archive, hands_name)
            frame_bboxes: dict[str, list[float]] = {}
            for side in ("left", "right"):
                hand = hands.get(side)
                if not isinstance(hand, dict):
                    continue
                visibility = _visibility(hand, stream_id)
                if visibility < min_visibility:
                    continue
                raw_box = (hand.get("boxes_amodal") or {}).get(stream_id)
                if raw_box is None:
                    continue
                frame_bboxes[side] = _bbox_to_xyxy(raw_box, bbox_format)
            if not frame_bboxes:
                continue

            frame_number = _frame_number(frame_id, sequence_index)
            image_path = image_dir / f"{frame_number:06d}.jpg"
            if overwrite or not image_path.exists():
                member = archive.extractfile(image_name)
                if member is None:
                    continue
                with image_path.open("wb") as output:
                    shutil.copyfileobj(member, output)

            frame_key = str(frame_number)
            row = {
                "metadata": {
                    "take_uid": take_uid,
                    "take_name": take_name,
                    "frame_number": frame_number,
                    "hot3d_frame_id": frame_id,
                    "source_tar": str(tar_path),
                    "stream_id": stream_id,
                },
                "left_hand_bbox": frame_bboxes.get("left", []),
                "right_hand_bbox": frame_bboxes.get("right", []),
                "left_hand_3d": [],
                "right_hand_3d": [],
                "left_hand_valid_3d": [],
                "right_hand_valid_3d": [],
            }
            take_anno[frame_key] = row
            manifest.append(
                {
                    "take_uid": take_uid,
                    "frame_number": frame_number,
                    "hot3d_frame_id": frame_id,
                    "image_path": str(image_path),
                    "hands_with_bbox": sorted(frame_bboxes),
                }
            )
            emitted += 1
    return take_anno, manifest


def _convert_tar_with_detector_tracks(
    tar_path: Path,
    *,
    tracks_path: Path,
    take_uid: str,
    take_name: str,
    image_dir: Path,
    stream_id: str,
    max_frames: int | None,
    overwrite: bool,
    min_score: float,
    expand: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    take_anno: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    tracks = _load_detector_track_boxes(tracks_path, min_score=min_score, expand=expand)
    with tarfile.open(tar_path) as archive:
        image_members = {
            _frame_id_from_name(name, f".image_{stream_id}.jpg"): name
            for name in archive.getnames()
            if name.endswith(f".image_{stream_id}.jpg")
        }
        if not image_members:
            raise ValueError(f"{tar_path} has no .image_{stream_id}.jpg entries")

        emitted = 0
        for sequence_index, frame_id in enumerate(sorted(image_members)):
            if max_frames is not None and emitted >= max_frames:
                break
            frame_number = _frame_number(frame_id, sequence_index)
            frame_bboxes = tracks.get(frame_number, {})
            if not frame_bboxes:
                continue

            image_path = image_dir / f"{frame_number:06d}.jpg"
            if overwrite or not image_path.exists():
                member = archive.extractfile(image_members[frame_id])
                if member is None:
                    continue
                with image_path.open("wb") as output:
                    shutil.copyfileobj(member, output)

            frame_key = str(frame_number)
            row = {
                "metadata": {
                    "take_uid": take_uid,
                    "take_name": take_name,
                    "frame_number": frame_number,
                    "hot3d_frame_id": frame_id,
                    "source_tar": str(tar_path),
                    "stream_id": stream_id,
                    "bbox_source": str(tracks_path),
                },
                "left_hand_bbox": frame_bboxes.get("left", []),
                "right_hand_bbox": frame_bboxes.get("right", []),
                "left_hand_3d": [],
                "right_hand_3d": [],
                "left_hand_valid_3d": [],
                "right_hand_valid_3d": [],
            }
            take_anno[frame_key] = row
            manifest.append(
                {
                    "take_uid": take_uid,
                    "frame_number": frame_number,
                    "hot3d_frame_id": frame_id,
                    "image_path": str(image_path),
                    "hands_with_bbox": sorted(frame_bboxes),
                    "bbox_source": str(tracks_path),
                }
            )
            emitted += 1
    return take_anno, manifest


def _load_detector_track_boxes(
    tracks_path: Path,
    *,
    min_score: float,
    expand: float,
) -> dict[int, dict[str, list[float]]]:
    raw_tracks = np.load(tracks_path, allow_pickle=True).item()
    by_frame: dict[int, dict[str, tuple[float, list[float]]]] = {}
    for entries in raw_tracks.values():
        for entry in entries:
            if not entry.get("det", True):
                continue
            box_values = np.asarray(entry.get("det_box"), dtype=float).reshape(-1)
            if box_values.size < 4:
                continue
            score = float(box_values[4]) if box_values.size >= 5 else 1.0
            if score < min_score:
                continue
            handedness = np.asarray(entry.get("det_handedness", [np.nan]), dtype=float)
            if handedness.size == 0 or not np.isfinite(handedness[0]):
                continue
            side = "left" if float(handedness[0]) < 0.5 else "right"
            frame_number = int(entry["frame"])
            box = _clip_xyxy(
                _expand_xyxy(box_values[:4], expand), width=1408, height=1408
            )
            current = by_frame.setdefault(frame_number, {}).get(side)
            if current is None or score > current[0]:
                by_frame[frame_number][side] = (score, box)
    return {
        frame: {side: box for side, (_score, box) in sides.items()}
        for frame, sides in by_frame.items()
    }


def _expand_xyxy(box: np.ndarray, factor: float) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    if factor <= 1.0:
        return [x1, y1, x2, y2]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = (x2 - x1) * factor * 0.5
    half_h = (y2 - y1) * factor * 0.5
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]


def _clip_xyxy(box: list[float], *, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]


def _resolve_tar_paths(
    root: Path, clips: list[str] | None, limit: int | None
) -> list[Path]:
    if root.is_file():
        paths = [root]
    elif clips:
        paths = []
        for clip in clips:
            candidate = root / f"{clip}.tar"
            if not candidate.exists():
                raise FileNotFoundError(f"missing HOT3D clip tar: {candidate}")
            paths.append(candidate)
    else:
        paths = sorted(root.glob("clip-*.tar"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no HOT3D clip tars found under {root}")
    return paths


def _read_json(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    member = archive.extractfile(name)
    if member is None:
        raise ValueError(f"missing tar member {name}")
    return json.load(member)


def _frame_id_from_name(name: str, suffix: str) -> str:
    return Path(name).name.removesuffix(suffix)


def _frame_number(frame_id: str, fallback: int) -> int:
    try:
        return int(frame_id)
    except ValueError:
        return fallback


def _visibility(hand: dict[str, Any], stream_id: str) -> float:
    modeled = hand.get("visibilities_modeled") or {}
    try:
        return float(modeled.get(stream_id, 1.0))
    except (TypeError, ValueError):
        return 1.0


def _bbox_to_xyxy(raw_box: Any, bbox_format: str) -> list[float]:
    if isinstance(raw_box, dict):
        if {"x1", "y1", "x2", "y2"} <= raw_box.keys():
            return [
                float(raw_box["x1"]),
                float(raw_box["y1"]),
                float(raw_box["x2"]),
                float(raw_box["y2"]),
            ]
        if {"xmin", "ymin", "xmax", "ymax"} <= raw_box.keys():
            return [
                float(raw_box["xmin"]),
                float(raw_box["ymin"]),
                float(raw_box["xmax"]),
                float(raw_box["ymax"]),
            ]
        if {"x", "y", "w", "h"} <= raw_box.keys():
            x, y, w, h = (
                float(raw_box["x"]),
                float(raw_box["y"]),
                float(raw_box["w"]),
                float(raw_box["h"]),
            )
            return [x, y, x + w, y + h]
    values = [float(value) for value in raw_box]
    if len(values) != 4:
        raise ValueError(f"expected bbox with four values, got {raw_box!r}")
    x0, y0, a, b = values
    if bbox_format == "xyxy":
        return [x0, y0, a, b]
    if bbox_format == "xywh":
        return [x0, y0, x0 + a, y0 + b]
    if a > x0 and b > y0:
        return [x0, y0, a, b]
    return [x0, y0, x0 + a, y0 + b]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bridge HOT3D-Clips train_aria tars to PCIE_EgoHandPose's "
            "Ego4D-style public test inference layout. The converter uses RGB "
            "frames and 2D hand boxes only; it does not copy HOT3D 3D hand poses "
            "into predictions."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common_prepare = argparse.ArgumentParser(add_help=False)
    common_prepare.add_argument("--hot3d-root", type=Path, required=True)
    common_prepare.add_argument("--clips", nargs="+")
    common_prepare.add_argument("--limit", type=int)
    common_prepare.add_argument("--stream-id", default=DEFAULT_STREAM_ID)

    inspect_parser = sub.add_parser("inspect", parents=[common_prepare])
    inspect_parser.set_defaults(func=inspect_hot3d)

    prepare = sub.add_parser("prepare", parents=[common_prepare])
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument(
        "--bbox-format", choices=("auto", "xyxy", "xywh"), default="xyxy"
    )
    prepare.add_argument("--min-visibility", type=float, default=0.0)
    prepare.add_argument("--max-frames-per-clip", type=int)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(func=prepare_hot3d_as_pcie_test)

    prepare_tracks = sub.add_parser("prepare-detector-tracks", parents=[common_prepare])
    prepare_tracks.add_argument("--output-root", type=Path, required=True)
    prepare_tracks.add_argument(
        "--tracks-root",
        type=Path,
        required=True,
        help=(
            "Root containing <clip>/rgb_<stream>/tracks_0_150/model_tracks.npy "
            "from a prior HaWoR detector/tracker run."
        ),
    )
    prepare_tracks.add_argument("--tracks-subdir", default="tracks_0_150")
    prepare_tracks.add_argument("--min-score", type=float, default=0.2)
    prepare_tracks.add_argument("--expand", type=float, default=1.0)
    prepare_tracks.add_argument("--max-frames-per-clip", type=int)
    prepare_tracks.add_argument("--overwrite", action="store_true")
    prepare_tracks.set_defaults(func=prepare_hot3d_detector_tracks_as_pcie_test)

    run = sub.add_parser("run")
    run.add_argument("--pcie-repo", type=Path, default=DEFAULT_PCIE_REPO)
    run.add_argument("--prepared-root", type=Path, required=True)
    run.add_argument("--pretrained-ckpt", type=Path, required=True)
    run.add_argument(
        "--cfg-file",
        type=Path,
        default=DEFAULT_PCIE_REPO / "configs" / "vit_base_transformerhead_joint.yaml",
    )
    run.add_argument("--output-dir", type=Path, required=True)
    run.set_defaults(func=run_pcie_inference)

    modal = sub.add_parser("print-modal-commands", parents=[common_prepare])
    modal.add_argument(
        "--pcie-repo", type=Path, default=Path("/cache/PCIE_EgoHandPose")
    )
    modal.add_argument(
        "--prepared-root", type=Path, default=Path("/cache/hot3d/pcie-smoke")
    )
    modal.add_argument(
        "--pretrained-ckpt",
        type=Path,
        default=Path("/cache/pcie/vitformer_base_ego4d.pth"),
    )
    modal.add_argument(
        "--cfg-file",
        type=Path,
        default=Path(
            "/cache/PCIE_EgoHandPose/configs/vit_base_transformerhead_joint.yaml"
        ),
    )
    modal.add_argument(
        "--output-dir", type=Path, default=Path("/cache/hot3d/pcie-output")
    )
    modal.set_defaults(func=print_modal_commands)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
