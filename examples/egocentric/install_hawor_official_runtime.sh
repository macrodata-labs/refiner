#!/usr/bin/env bash
set -euo pipefail

HAWOR_ROOT="${HAWOR_ROOT:-/opt/HaWoR}"
HAWOR_REPO="${HAWOR_REPO:-https://github.com/ThunderVVV/HaWoR.git}"
HAWOR_REF="${HAWOR_REF:-main}"
REFINER_EXPORT="${REFINER_EXPORT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hawor_refiner_export.py}"
HAWOR_MANO_REPO="${HAWOR_MANO_REPO:-macrodata/hawor-official-assets}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

if [ ! -d "$HAWOR_ROOT/.git" ]; then
  rm -rf "$HAWOR_ROOT"
  git clone --recursive "$HAWOR_REPO" "$HAWOR_ROOT"
fi

git -C "$HAWOR_ROOT" fetch --depth 1 origin "$HAWOR_REF" || true
git -C "$HAWOR_ROOT" checkout "$HAWOR_REF"
git -C "$HAWOR_ROOT" reset --hard "$HAWOR_REF"
git -C "$HAWOR_ROOT" submodule update --init --recursive

python -m pip install --upgrade pip

if [ "${HAWOR_FORCE_TORCH_INSTALL:-0}" = "1" ] || ! python - <<'PY'
import torch
PY
then
  python -m pip install \
    ${HAWOR_FORCE_TORCH_INSTALL:+--force-reinstall} \
    ${HAWOR_TORCH_PACKAGES:-torch torchvision} \
    --index-url "${HAWOR_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
fi

python -m pip install wheel ninja
python - <<'PY' "$HAWOR_ROOT/requirements.txt" /tmp/hawor-requirements-runtime.txt
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:])
skip = ("torch-scatter",)
if __import__("os").environ.get("HAWOR_SKIP_RENDERER") == "1":
    skip = (*skip, "pytorch3d")
lines = [
    line
    for line in source.read_text().splitlines()
    if line.strip() and not any(token in line for token in skip)
]
output.write_text("\n".join(lines) + "\n")
PY
python -m pip install --no-build-isolation -r /tmp/hawor-requirements-runtime.txt
if [ "${HAWOR_FORCE_TORCH_INSTALL:-0}" = "1" ]; then
  python -m pip install \
    --force-reinstall \
    ${HAWOR_TORCH_PACKAGES:-torch torchvision} \
    --index-url "${HAWOR_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
fi
TORCH_BASE_VERSION="$(python - <<'PY'
import torch
print(torch.__version__.split("+", 1)[0])
PY
)"
TORCH_CUDA_TAG="$(python - <<'PY'
import torch
cuda = torch.version.cuda or "12.4"
print("cu" + cuda.replace(".", ""))
PY
)"
python -m pip install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_BASE_VERSION}+${TORCH_CUDA_TAG}.html"
python -m pip install pytorch-lightning==2.2.4 --no-deps
python -m pip install lightning-utilities torchmetrics==1.4.0
python -m pip install huggingface-hub scipy imageio-ffmpeg requests
python - <<'PY'
from pathlib import Path
import imageio_ffmpeg
import os

source = Path(imageio_ffmpeg.get_ffmpeg_exe())
target = Path("/usr/local/bin/ffmpeg")
if not target.exists():
    target.symlink_to(source)
os.chmod(source, 0o755)
PY

if [ "${HAWOR_SKIP_RENDERER:-0}" != "1" ]; then
  (
    cd "$HAWOR_ROOT"
    python - <<'PY'
import pytorch3d
from lib.vis.renderer import Renderer

version = getattr(pytorch3d, "__version__", "unknown")
print(f"PyTorch3D renderer available: {version}; {Renderer.__name__}")
PY
  )
fi

(
  cd "$HAWOR_ROOT/thirdparty/DROID-SLAM"
  rm -rf build dist droid_backends.egg-info
  rm -f droid_backends*.so thirdparty/lietorch/lietorch_backends*.so
  python - <<'PY'
from pathlib import Path

path = Path("setup.py")
patched = []
for line in path.read_text().splitlines():
    if "-gencode=arch=compute_" in line:
        continue
    patched.append(line)
    if "'nvcc': ['-O3'," in line or "'nvcc': ['-O2'," in line:
        patched.append("                    '-gencode=arch=compute_90,code=sm_90',")
path.write_text("\n".join(patched) + "\n")

for source in (
    Path("src/altcorr_kernel.cu"),
    Path("src/correlation_kernels.cu"),
):
    text = source.read_text()
    text = text.replace("fmap1.type()", "fmap1.scalar_type()")
    text = text.replace("volume.type()", "volume.scalar_type()")
    source.write_text(text)

for source in Path("thirdparty/lietorch/lietorch").rglob("*"):
    if source.suffix not in {".cpp", ".cu", ".h"}:
        continue
    text = source.read_text()
    text = text.replace(".type()", ".scalar_type()")
    text = text.replace(".device().scalar_type()", ".device().type()")
    source.write_text(text)
PY
  if ! python setup.py build_ext --inplace > /tmp/droid-slam-build.log 2>&1; then
    tail -400 /tmp/droid-slam-build.log >&2
    exit 1
  fi
  python - <<'PY'
import torch  # noqa: F401
import droid_backends

doc = droid_backends.ba.__doc__ or ""
expected = "arg14: bool"
unexpected = "arg27: float"
if expected not in doc or unexpected in doc:
    raise RuntimeError(
        "DROID-SLAM droid_backends.ba ABI mismatch. Expected the local "
        f"15-argument binding, got: {doc}"
    )
print("DROID-SLAM droid_backends.ba ABI verified")
PY
)

mkdir -p \
  "$HAWOR_ROOT/weights/external" \
  "$HAWOR_ROOT/weights/hawor/checkpoints" \
  "$HAWOR_ROOT/weights/hawor" \
  "$HAWOR_ROOT/thirdparty/Metric3D/weights" \
  "$HAWOR_ROOT/_DATA/data/mano" \
  "$HAWOR_ROOT/_DATA/data_left/mano_left"

download_if_missing() {
  local url="$1"
  local output="$2"
  if [ ! -s "$output" ]; then
    python - <<'PY' "$url" "$output"
from pathlib import Path
import requests
import sys
import time

url, output = sys.argv[1], Path(sys.argv[2])
output.parent.mkdir(parents=True, exist_ok=True)
partial = output.with_suffix(output.suffix + ".partial")
last_error = None
for attempt in range(1, 11):
    headers = {}
    if partial.exists():
        headers["Range"] = f"bytes={partial.stat().st_size}-"
    try:
        with requests.get(url, headers=headers, stream=True, timeout=(30, 600)) as response:
            mode = "ab" if response.status_code == 206 and partial.exists() else "wb"
            response.raise_for_status()
            with partial.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        partial.replace(output)
        break
    except Exception as exc:
        last_error = exc
        print(f"download failed attempt={attempt} url={url}: {exc}", flush=True)
        time.sleep(min(60, attempt * 5))
else:
    raise RuntimeError(f"failed to download {url}: {last_error}")
PY
  fi
}

download_if_missing \
  "https://huggingface.co/ThunderVVV/HaWoR/resolve/main/external/droid.pth" \
  "$HAWOR_ROOT/weights/external/droid.pth"
download_if_missing \
  "https://huggingface.co/ThunderVVV/HaWoR/resolve/main/external/metric_depth_vit_large_800k.pth" \
  "$HAWOR_ROOT/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth"
download_if_missing \
  "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt" \
  "$HAWOR_ROOT/weights/external/detector.pt"
download_if_missing \
  "https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt" \
  "$HAWOR_ROOT/weights/hawor/checkpoints/hawor.ckpt"
download_if_missing \
  "https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt" \
  "$HAWOR_ROOT/weights/hawor/checkpoints/infiller.pt"
download_if_missing \
  "https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml" \
  "$HAWOR_ROOT/weights/hawor/model_config.yaml"

python - <<'PY' "$HAWOR_MANO_REPO" "$HAWOR_ROOT"
from huggingface_hub import hf_hub_download
from pathlib import Path
import os
import shutil
import sys

repo_id, root = sys.argv[1], Path(sys.argv[2])
token = os.environ.get("HF_TOKEN")
right = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="mano/MANO_RIGHT.pkl", token=token)
left = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="mano/MANO_LEFT.pkl", token=token)
shutil.copyfile(right, root / "_DATA/data/mano/MANO_RIGHT.pkl")
shutil.copyfile(left, root / "_DATA/data_left/mano_left/MANO_LEFT.pkl")
PY

cp "$REFINER_EXPORT" "$HAWOR_ROOT/refiner_export.py"

missing=0
for required in \
  "$HAWOR_ROOT/weights/external/droid.pth" \
  "$HAWOR_ROOT/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth" \
  "$HAWOR_ROOT/weights/external/detector.pt" \
  "$HAWOR_ROOT/weights/hawor/checkpoints/hawor.ckpt" \
  "$HAWOR_ROOT/weights/hawor/checkpoints/infiller.pt" \
  "$HAWOR_ROOT/weights/hawor/model_config.yaml" \
  "$HAWOR_ROOT/_DATA/data/mano/MANO_RIGHT.pkl" \
  "$HAWOR_ROOT/_DATA/data_left/mano_left/MANO_LEFT.pkl" \
  "$HAWOR_ROOT/refiner_export.py"; do
  if [ ! -s "$required" ]; then
    echo "missing required HaWoR runtime file: $required" >&2
    missing=1
  fi
done

if [ "$missing" -ne 0 ]; then
  exit 2
fi

echo "Official HaWoR runtime installed at $HAWOR_ROOT"
