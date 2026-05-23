from __future__ import annotations

import argparse
import json
import os
import sys
import types
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
    sys.path.insert(0, str(droid_root / "thirdparty" / "lietorch"))
    sys.path.insert(0, str(droid_root / "droid_slam"))
    sys.path.insert(0, str(droid_root))
    sys.path.insert(0, str(hawor_root))
    return hawor_root


HAWOR_ROOT = _add_hawor_to_path()

if os.environ.get("HAWOR_SKIP_RENDERER") == "1":
    renderer_module = types.ModuleType("lib.vis.renderer")

    class Renderer:  # noqa: D101
        def __init__(self, width, height, focal_length, device, **kwargs):
            self.width = int(width)
            self.height = int(height)
            self.focal_length = float(focal_length)

        def create_camera_from_cv(self, *args, **kwargs):
            return None, None

        def render_multiple(self, verts_list, faces, colors_list, cameras, lights):
            verts = torch.as_tensor(verts_list).detach().cpu().numpy().reshape(-1, 3)
            valid = np.isfinite(verts).all(axis=1) & (np.abs(verts[:, 2]) > 1e-6)
            mask = np.zeros((self.height, self.width), dtype=bool)
            if valid.any():
                points = verts[valid]
                u = self.focal_length * points[:, 0] / points[:, 2] + self.width / 2
                v = self.focal_length * points[:, 1] / points[:, 2] + self.height / 2
                finite = np.isfinite(u) & np.isfinite(v)
                if finite.any():
                    u = np.clip(u[finite], 0, self.width - 1)
                    v = np.clip(v[finite], 0, self.height - 1)
                    pad = 8
                    x0 = max(0, int(np.floor(u.min())) - pad)
                    x1 = min(self.width, int(np.ceil(u.max())) + pad + 1)
                    y0 = max(0, int(np.floor(v.min())) - pad)
                    y1 = min(self.height, int(np.ceil(v.max())) + pad + 1)
                    mask[y0:y1, x0:x1] = True
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return image, mask

    renderer_module.Renderer = Renderer
    sys.modules["lib.vis.renderer"] = renderer_module

from hawor.utils.process import run_mano, run_mano_left  # noqa: E402
from hawor.utils.rotation import angle_axis_to_rotation_matrix  # noqa: E402
from scripts.scripts_test_video.detect_track_video import detect_track_video  # noqa: E402
from scripts.scripts_test_video.hawor_slam import hawor_slam  # noqa: E402
import scripts.scripts_test_video.hawor_video as hawor_video_module  # noqa: E402
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

    hawor_video_module.load_slam_cam = load_slam_cam
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
            camera_t_world_camera=camera["T_world_camera"],
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
    camera_t_world_camera: list,
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
    t_world_camera = np.asarray(camera_t_world_camera, dtype=np.float64)
    t_camera_world = np.linalg.inv(t_world_camera)
    t_camera_wrist = np.matmul(t_camera_world, t_world_wrist)
    joints_camera = _transform_points(t_camera_world, joints)
    return {
        "T_camera_wrist": t_camera_wrist.tolist(),
        "T_world_wrist": t_world_wrist.tolist(),
        "mano_pose": _to_numpy(hand_pose).tolist(),
        "mano_shape": _to_numpy(betas).tolist(),
        "joints_camera": joints_camera.tolist(),
        "joints_world": joints.tolist(),
        "confidence": _to_numpy(valid).astype(float).tolist(),
    }


def _transform_points(transforms: np.ndarray, points: np.ndarray) -> np.ndarray:
    rotation = transforms[:, :3, :3]
    translation = transforms[:, :3, 3]
    return points @ np.swapaxes(rotation, 1, 2) + translation[:, None, :]


def _transforms_from_rt(
    rotation: torch.Tensor, translation: torch.Tensor
) -> np.ndarray:
    rotation_np = _to_numpy(rotation)
    translation_np = _to_numpy(translation)
    transforms = np.tile(np.eye(4, dtype=np.float64), (len(rotation_np), 1, 1))
    transforms[:, :3, :3] = rotation_np
    transforms[:, :3, 3] = translation_np
    return transforms


def load_slam_cam(fpath: str):
    print(f"Loading cameras from {fpath}...")
    pred_cam = dict(np.load(fpath, allow_pickle=True))
    pred_traj = np.asarray(pred_cam["traj"], dtype=np.float32)
    scale = np.asarray(pred_cam["scale"], dtype=np.float32)
    t_c2w_sla = torch.tensor(pred_traj[:, :3], dtype=torch.float32) * torch.tensor(
        scale,
        dtype=torch.float32,
    )
    pred_camq = torch.tensor(pred_traj[:, 3:], dtype=torch.float32)
    r_c2w_sla = _quaternion_to_matrix(pred_camq[:, [3, 0, 1, 2]])
    r_w2c_sla = r_c2w_sla.transpose(-1, -2)
    t_w2c_sla = -torch.einsum("bij,bj->bi", r_w2c_sla, t_c2w_sla)
    return r_w2c_sla, t_w2c_sla, r_c2w_sla, t_c2w_sla


def _quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    values = (
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j),
    )
    return torch.stack(values, -1).reshape(quaternions.shape[:-1] + (3, 3))


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
