from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent

from huggingface_hub import snapshot_download


RUNTIME_MODULE = r"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from scipy import sparse
from smplx.utils import Struct


def safetensors_dir() -> Path:
    return Path(os.environ.get("HAWOR_SAFETENSORS_DIR", "weights/safetensors"))


def asset_path(name: str) -> Path:
    path = Path(name)
    if path.suffix == ".safetensors" and path.exists():
        return path
    by_stem = {
        "droid": "droid.safetensors",
        "droid.pth": "droid.safetensors",
        "metric3d": "metric3d.safetensors",
        "metric_depth_vit_large_800k.pth": "metric3d.safetensors",
        "MANO_RIGHT.pkl": "MANO_RIGHT.safetensors",
        "MANO_LEFT.pkl": "MANO_LEFT.safetensors",
    }
    return safetensors_dir() / by_stem.get(path.name, path.name)


def load_state_dict_safetensors(path: str | os.PathLike[str]) -> dict[str, torch.Tensor]:
    tensors = load_file(str(asset_path(str(path))))
    return dict(tensors)


def load_droid_state_dict(path: str | os.PathLike[str]) -> dict[str, torch.Tensor]:
    return load_state_dict_safetensors(path)


def load_metric3d_checkpoint(path: str | os.PathLike[str]) -> dict[str, dict[str, torch.Tensor]]:
    tensors = load_state_dict_safetensors(path)
    prefix = "model_state_dict."
    state_dict = {
        name[len(prefix):] if name.startswith(prefix) else name: tensor
        for name, tensor in tensors.items()
    }
    return {"model_state_dict": state_dict}


def _manifest(asset_dir: Path) -> dict[str, Any]:
    return json.loads((asset_dir / "manifest.json").read_text())


def _manifest_asset(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    for asset in manifest["assets"]:
        if asset["name"] == name:
            return asset
    raise KeyError(f"manifest has no asset named {name!r}")


def _torch_to_numpy(value: torch.Tensor) -> Any:
    array = value.detach().cpu().numpy()
    return array.item() if array.shape == () else array


def _metadata_value(entry: dict[str, Any]) -> Any:
    value = entry.get("repr")
    if value is None:
        return None
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    return value


def load_mano_struct(model_path: str | os.PathLike[str], *, is_rhand: bool = True) -> Struct:
    requested = Path(model_path)
    name = "MANO_RIGHT" if is_rhand else "MANO_LEFT"
    if requested.suffix == ".safetensors":
        name = requested.stem
        tensor_path = requested
    else:
        tensor_path = asset_path("MANO_RIGHT.pkl" if is_rhand else "MANO_LEFT.pkl")

    asset_dir = tensor_path.parent
    manifest = _manifest(asset_dir)
    asset = _manifest_asset(manifest, name)
    tensors = load_file(str(tensor_path))
    payload: dict[str, Any] = {}

    sparse_names = {
        key for key in asset.get("metadata", {}) if f"{key}.data" in tensors
    }
    for key in sparse_names:
        metadata = asset["metadata"][key]
        payload[key] = sparse.csc_matrix(
            (
                _torch_to_numpy(tensors[f"{key}.data"]),
                _torch_to_numpy(tensors[f"{key}.indices"]),
                _torch_to_numpy(tensors[f"{key}.indptr"]),
            ),
            shape=tuple(metadata["shape"]),
        )

    sparse_parts = {
        part
        for key in sparse_names
        for part in (f"{key}.data", f"{key}.indices", f"{key}.indptr")
    }
    for key, tensor in tensors.items():
        if key not in sparse_parts:
            payload[key] = _torch_to_numpy(tensor)

    for key, entry in asset.get("metadata", {}).items():
        if key not in payload and key not in sparse_names:
            payload[key] = _metadata_value(entry)

    return Struct(**payload)
"""


PATCHES = [
    (
        "thirdparty/DROID-SLAM/droid_slam/droid.py",
        "from collections import OrderedDict\n",
        "from collections import OrderedDict\nfrom refiner_safetensors_runtime import load_droid_state_dict\n",
    ),
    (
        "thirdparty/DROID-SLAM/droid_slam/droid.py",
        "torch.load(weights).items()])",
        "load_droid_state_dict(weights).items()])",
    ),
    (
        "lib/pipeline/masked_droid_slam.py",
        'parser.add_argument("--weights", default="weights/external/droid.pth")',
        'parser.add_argument("--weights", default="weights/safetensors/droid.safetensors")',
    ),
    (
        "scripts/scripts_test_video/hawor_slam.py",
        "Metric3D('thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth')",
        "Metric3D('weights/safetensors/metric3d.safetensors')",
    ),
    (
        "thirdparty/Metric3D/mono/utils/running.py",
        "import glob\n",
        "import glob\nfrom refiner_safetensors_runtime import load_metric3d_checkpoint\n",
    ),
    (
        "thirdparty/Metric3D/mono/utils/running.py",
        'checkpoint = torch.load(load_path, map_location="cpu")',
        "checkpoint = load_metric3d_checkpoint(load_path)",
    ),
    (
        "lib/models/mano_wrapper.py",
        "from smplx.vertex_ids import vertex_ids\n",
        "from smplx.vertex_ids import vertex_ids\nfrom refiner_safetensors_runtime import load_mano_struct\n",
    ),
    (
        "lib/models/mano_wrapper.py",
        "super(MANO, self).__init__(*args, **kwargs)",
        "kwargs.setdefault('data_struct', load_mano_struct(kwargs.get('model_path', args[0] if args else ''), is_rhand=kwargs.get('is_rhand', True)))\n        super(MANO, self).__init__(*args, **kwargs)",
    ),
    (
        "infiller/hand_utils/mano_wrapper.py",
        "from smplx.vertex_ids import vertex_ids\n",
        "from smplx.vertex_ids import vertex_ids\nfrom refiner_safetensors_runtime import load_mano_struct\n",
    ),
    (
        "infiller/hand_utils/mano_wrapper.py",
        "super(MANO, self).__init__(*args, **kwargs)",
        "kwargs.setdefault('data_struct', load_mano_struct(kwargs.get('model_path', args[0] if args else ''), is_rhand=kwargs.get('is_rhand', True)))\n        super(MANO, self).__init__(*args, **kwargs)",
    ),
]


def patch_text(path: Path, before: str, after: str) -> None:
    text = path.read_text()
    if after in text:
        return
    if before not in text:
        raise RuntimeError(f"patch anchor not found in {path}: {before!r}")
    path.write_text(text.replace(before, after, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch HaWoR to load Macrodata safetensors assets natively."
    )
    parser.add_argument("--hawor-root", type=Path, default=Path("/opt/HaWoR"))
    parser.add_argument("--repo-id", default="macrodata/egovision-safetensors")
    parser.add_argument("--repo-type", default="model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.hawor_root
    asset_dir = root / "weights/safetensors"
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=asset_dir,
        allow_patterns=["*.safetensors", "manifest.json"],
    )

    (root / "refiner_safetensors_runtime.py").write_text(
        dedent(RUNTIME_MODULE).lstrip()
    )
    for relative_path, before, after in PATCHES:
        patch_text(root / relative_path, before, after)

    print(
        json.dumps(
            {"hawor_root": str(root), "safetensors_dir": str(asset_dir)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
