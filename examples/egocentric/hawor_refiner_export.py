from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

_TORCH_LOAD = torch.load


def _torch_load_trusted_checkpoint(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_trusted_checkpoint


def _add_hawor_to_path() -> Path:
    script_path = Path(__file__).resolve()
    hawor_root = script_path.parent
    droid_root = hawor_root / "thirdparty" / "DROID-SLAM"
    sys.path.insert(0, str(droid_root / "droid_slam"))
    sys.path.insert(0, str(droid_root))
    sys.path.insert(0, str(hawor_root))
    return hawor_root


HAWOR_ROOT = _add_hawor_to_path()

from hawor.utils.process import run_mano, run_mano_left  # noqa: E402
from hawor.utils.rotation import angle_axis_to_rotation_matrix  # noqa: E402
from lib.eval_utils.custom_utils import load_slam_cam  # noqa: E402
from scripts.scripts_test_video.detect_track_video import detect_track_video  # noqa: E402
from scripts.scripts_test_video.hawor_slam import hawor_slam  # noqa: E402
from scripts.scripts_test_video.hawor_video import (  # noqa: E402
    hawor_infiller,
    hawor_motion_estimation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "--video_path", dest="video_path", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--input_type", default="file")
    parser.add_argument(
        "--checkpoint",
        default=str(HAWOR_ROOT / "weights/hawor/checkpoints/hawor.ckpt"),
    )
    parser.add_argument(
        "--infiller_weight",
        default=str(HAWOR_ROOT / "weights/hawor/checkpoints/infiller.pt"),
    )
    args = parser.parse_args()

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    frame_chunks_all, img_focal = hawor_motion_estimation(
        args,
        start_idx,
        end_idx,
        seq_folder,
    )
    slam_path = Path(seq_folder) / f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
    if not slam_path.exists():
        hawor_slam(args, start_idx, end_idx)
    _, _, r_c2w, t_c2w = load_slam_cam(str(slam_path))
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
        args,
        start_idx,
        end_idx,
        frame_chunks_all,
    )

    payload = _build_payload(
        args=args,
        imgfiles=list(imgfiles),
        img_focal=float(img_focal),
        r_c2w=r_c2w,
        t_c2w=t_c2w,
        pred_trans=pred_trans,
        pred_rot=pred_rot,
        pred_hand_pose=pred_hand_pose,
        pred_betas=pred_betas,
        pred_valid=pred_valid,
    )

    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload), encoding="utf-8")


def _build_payload(
    *,
    args: argparse.Namespace,
    imgfiles: list[str],
    img_focal: float,
    r_c2w: torch.Tensor,
    t_c2w: torch.Tensor,
    pred_trans: torch.Tensor,
    pred_rot: torch.Tensor,
    pred_hand_pose: torch.Tensor,
    pred_betas: torch.Tensor,
    pred_valid: torch.Tensor,
) -> dict[str, Any]:
    timestamps = _timestamps(args.video_path, len(imgfiles))
    camera = {
        "T_world_camera": _transforms_from_rt(r_c2w, t_c2w).tolist(),
        "units": "meters",
        "source": "hawor_adaptive_droid_slam_metric3d",
    }
    payload: dict[str, Any] = {
        "timestamps": timestamps,
        "camera": camera,
        "metadata": {
            "provider": "hawor",
            "video_path": args.video_path,
            "img_focal": img_focal,
        },
    }

    for hand_index, side in enumerate(("left", "right")):
        payload[f"{side}_hand"] = _hand_payload(
            side=side,
            trans=pred_trans[hand_index],
            rot=pred_rot[hand_index],
            hand_pose=pred_hand_pose[hand_index],
            betas=pred_betas[hand_index],
            valid=pred_valid[hand_index],
        )
    return payload


def _hand_payload(
    *,
    side: str,
    trans: torch.Tensor,
    rot: torch.Tensor,
    hand_pose: torch.Tensor,
    betas: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, Any]:
    trans_t = torch.as_tensor(trans).cpu()
    rot_t = torch.as_tensor(rot).cpu()
    hand_pose_t = torch.as_tensor(hand_pose).cpu()
    betas_t = torch.as_tensor(betas).cpu()
    rot_mats = angle_axis_to_rotation_matrix(rot_t).cpu()
    t_world_wrist = _transforms_from_rt(rot_mats, trans_t)
    mano_fn = run_mano_left if side == "left" else run_mano
    mano = mano_fn(
        trans_t[None],
        rot_t[None],
        hand_pose_t[None],
        betas=betas_t[None],
    )
    joints = mano["joints"][0].detach().cpu().numpy()
    return {
        "T_world_wrist": t_world_wrist.tolist(),
        "mano_pose": _to_numpy(hand_pose).tolist(),
        "mano_shape": _to_numpy(betas).tolist(),
        "joints_world": joints.tolist(),
        "confidence": _to_numpy(valid).astype(float).tolist(),
    }


def _transforms_from_rt(
    rotation: torch.Tensor, translation: torch.Tensor
) -> np.ndarray:
    rotation_np = _to_numpy(rotation)
    translation_np = _to_numpy(translation)
    transforms = np.tile(np.eye(4, dtype=np.float64), (len(rotation_np), 1, 1))
    transforms[:, :3, :3] = rotation_np
    transforms[:, :3, 3] = translation_np
    return transforms


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _timestamps(video_path: str, frame_count: int) -> list[float]:
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if fps is None or fps <= 0:
        fps = 30.0
    return [index / float(fps) for index in range(frame_count)]


if __name__ == "__main__":
    os.chdir(HAWOR_ROOT)
    main()
