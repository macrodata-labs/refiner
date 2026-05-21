from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import gdown
from huggingface_hub import create_repo, hf_hub_download, upload_folder
from safetensors.torch import save_file


PUBLIC_WEIGHTS = {
    "detector": {
        "repo_id": "rolpotamias/WiLoR",
        "filename": "pretrained_models/detector.pt",
        "repo_type": "space",
    },
    "droid": {
        "gdrive_id": "1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh",
        "filename": "droid.pth",
        "source": "princeton-vl/DROID-SLAM",
    },
    "metric3d": {
        "repo_id": "JUGGHM/Metric3D",
        "filename": "metric_depth_vit_large_800k.pth",
        "repo_type": "model",
    },
    "hawor": {
        "repo_id": "ThunderVVV/HaWoR",
        "filename": "hawor/checkpoints/hawor.ckpt",
        "repo_type": "space",
    },
    "infiller": {
        "repo_id": "ThunderVVV/HaWoR",
        "filename": "hawor/checkpoints/infiller.pt",
        "repo_type": "space",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path: Path) -> Any:
    for alias, value in {
        "bool": bool,
        "complex": complex,
        "float": float,
        "int": int,
        "object": object,
        "str": str,
    }.items():
        if not hasattr(np, alias):
            setattr(np, alias, value)
    install_chumpy_pickle_shim()
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def install_chumpy_pickle_shim() -> None:
    try:
        import chumpy  # noqa: F401

        return
    except ImportError:
        pass

    class CompatCh:
        def __setstate__(self, state: Any) -> None:
            self.__dict__.update(state if isinstance(state, dict) else {"state": state})

        @property
        def r(self) -> Any:
            return self.__dict__.get("x")

    class CompatSelect(CompatCh):
        @property
        def r(self) -> Any:
            source = getattr(self.__dict__.get("a"), "r", None)
            idxs = self.__dict__.get("idxs")
            shape = self.__dict__.get("preferred_shape")
            if source is None or idxs is None:
                return None
            selected = np.asarray(source).reshape(-1)[idxs]
            return selected.reshape(shape) if shape is not None else selected

    chumpy = types.ModuleType("chumpy")
    chumpy_ch = types.ModuleType("chumpy.ch")
    chumpy_reordering = types.ModuleType("chumpy.reordering")
    chumpy_ch.Ch = CompatCh
    chumpy_reordering.Select = CompatSelect
    chumpy.ch = chumpy_ch
    chumpy.reordering = chumpy_reordering
    sys.modules.update(
        {
            "chumpy": chumpy,
            "chumpy.ch": chumpy_ch,
            "chumpy.reordering": chumpy_reordering,
        }
    )


def to_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray) and value.dtype != object:
        return torch.as_tensor(value)
    if (
        hasattr(value, "r")
        and isinstance(value.r, np.ndarray)
        and value.r.dtype != object
    ):
        return torch.as_tensor(value.r)
    if isinstance(value, (bool, int, float, complex, np.number, np.bool_)):
        return torch.as_tensor(value)
    if all(hasattr(value, attr) for attr in ("data", "indices", "indptr", "shape")):
        return None
    return None


def flatten_tensors(
    value: Any, prefix: str = ""
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, Any] = {}

    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            child_tensors, child_metadata = flatten_tensors(item, name)
            tensors.update(child_tensors)
            metadata.update(child_metadata)
        return tensors, metadata

    tensor = to_tensor(value)
    if tensor is not None:
        tensors[prefix] = tensor.clone().contiguous()
        return tensors, metadata

    if all(hasattr(value, attr) for attr in ("data", "indices", "indptr", "shape")):
        sparse_prefix = prefix
        tensors[f"{sparse_prefix}.data"] = (
            torch.as_tensor(value.data).clone().contiguous()
        )
        tensors[f"{sparse_prefix}.indices"] = (
            torch.as_tensor(value.indices).clone().contiguous()
        )
        tensors[f"{sparse_prefix}.indptr"] = (
            torch.as_tensor(value.indptr).clone().contiguous()
        )
        metadata[sparse_prefix] = {
            "kind": type(value).__name__,
            "shape": list(value.shape),
        }
        return tensors, metadata

    metadata[prefix] = {
        "kind": type(value).__name__,
        "repr": repr(value)[:500],
    }
    return tensors, metadata


def extract_state_dict(checkpoint: Any) -> Any:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "net", "module"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                return candidate
    return checkpoint


def convert_checkpoint(
    source_path: Path,
    output_path: Path,
    *,
    source: dict[str, Any],
    pickle_asset: bool = False,
) -> dict[str, Any]:
    if pickle_asset:
        loaded = load_pickle(source_path)
    else:
        loaded = torch.load(source_path, map_location="cpu")
        loaded = extract_state_dict(loaded)

    tensors, extra_metadata = flatten_tensors(loaded)
    if not tensors:
        raise ValueError(f"no tensors found in {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path))

    return {
        "name": output_path.stem,
        "source": source,
        "source_sha256": sha256_file(source_path),
        "safetensors": str(output_path.name),
        "safetensors_sha256": sha256_file(output_path),
        "tensor_count": len(tensors),
        "tensors": {
            name: {
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": list(tensor.shape),
            }
            for name, tensor in sorted(tensors.items())
        },
        "metadata": extra_metadata,
    }


def convert_public_weight(name: str, output_dir: Path) -> dict[str, Any]:
    source = PUBLIC_WEIGHTS[name]
    if "gdrive_id" in source:
        downloaded = output_dir / "original" / str(source["filename"])
        if not downloaded.exists():
            downloaded.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(
                id=str(source["gdrive_id"]), output=str(downloaded), quiet=False
            )
    else:
        downloaded = Path(hf_hub_download(**source))
    output_path = output_dir / f"{name}.safetensors"
    return convert_checkpoint(downloaded, output_path, source=source)


def convert_mano(path: Path, output_dir: Path, name: str) -> dict[str, Any]:
    output_path = output_dir / f"{name}.safetensors"
    return convert_checkpoint(
        path,
        output_path,
        source={"path": str(path), "asset": "MANO", "licensed": True},
        pickle_asset=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HaWoR public checkpoints and optional licensed MANO files to safetensors."
    )
    parser.add_argument("--output-dir", default="artifacts/hawor-safetensors")
    parser.add_argument(
        "--weights",
        nargs="+",
        default=list(PUBLIC_WEIGHTS),
        choices=sorted(PUBLIC_WEIGHTS),
    )
    parser.add_argument("--mano-right", type=Path)
    parser.add_argument("--mano-left", type=Path)
    parser.add_argument(
        "--upload-repo",
        help="Optional Hugging Face repo id to upload to, for example macrodata/hawor-safetensors.",
    )
    parser.add_argument(
        "--repo-type", default="dataset", choices=["dataset", "model", "space"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "format": "hawor-safetensors-v1",
        "assets": [],
        "notes": [
            "MANO conversion only includes user-provided licensed MANO files.",
            "HaWoR runtime loaders still need to be patched before these safetensors can replace original files.",
        ],
    }

    for name in args.weights:
        manifest["assets"].append(convert_public_weight(name, output_dir))

    if args.mano_right is not None:
        manifest["assets"].append(
            convert_mano(args.mano_right, output_dir, "MANO_RIGHT")
        )
    if args.mano_left is not None:
        manifest["assets"].append(convert_mano(args.mano_left, output_dir, "MANO_LEFT"))

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if args.upload_repo:
        create_repo(args.upload_repo, repo_type=args.repo_type, exist_ok=True)
        upload_folder(
            repo_id=args.upload_repo,
            repo_type=args.repo_type,
            folder_path=str(output_dir),
            commit_message="upload hawor safetensors assets",
        )

    print(
        json.dumps(
            {"output_dir": str(output_dir), "manifest": str(manifest_path)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
