#!/usr/bin/env bash
set -euo pipefail

MEGASAM_ROOT="${MEGASAM_ROOT:-/opt/mega-sam}"
MEGASAM_REPO="${MEGASAM_REPO:-https://github.com/mega-sam/mega-sam.git}"
MEGASAM_REF="${MEGASAM_REF:-main}"
LINGBOT_DEPTH_ROOT="${LINGBOT_DEPTH_ROOT:-/opt/lingbot-depth}"
LINGBOT_DEPTH_REPO="${LINGBOT_DEPTH_REPO:-https://github.com/Robbyant/lingbot-depth.git}"
LINGBOT_DEPTH_REF="${LINGBOT_DEPTH_REF:-main}"
REFINER_EXPORT="${REFINER_EXPORT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/megasam_refiner_export.py}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

if [ ! -d "$MEGASAM_ROOT/.git" ]; then
  rm -rf "$MEGASAM_ROOT"
  echo "[refiner-megasam] cloning $MEGASAM_REPO"
  git clone --recursive "$MEGASAM_REPO" "$MEGASAM_ROOT"
fi

echo "[refiner-megasam] checking out $MEGASAM_REF"
git -C "$MEGASAM_ROOT" fetch --depth 1 origin "$MEGASAM_REF" || true
git -C "$MEGASAM_ROOT" checkout "$MEGASAM_REF"
git -C "$MEGASAM_ROOT" submodule update --init --recursive

if [ ! -d "$LINGBOT_DEPTH_ROOT/.git" ]; then
  rm -rf "$LINGBOT_DEPTH_ROOT"
  echo "[refiner-megasam] cloning $LINGBOT_DEPTH_REPO"
  git clone "$LINGBOT_DEPTH_REPO" "$LINGBOT_DEPTH_ROOT"
fi

echo "[refiner-megasam] checking out LingBot-Depth $LINGBOT_DEPTH_REF"
git -C "$LINGBOT_DEPTH_ROOT" fetch --depth 1 origin "$LINGBOT_DEPTH_REF" || true
git -C "$LINGBOT_DEPTH_ROOT" checkout "$LINGBOT_DEPTH_REF"

echo "[refiner-megasam] patching UniDepth xformers compatibility"
python - <<'PY' "$MEGASAM_ROOT"
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / "UniDepth" / "unidepth" / "layers" / "nystrom_attention.py"
path.write_text(
    """from .attention import AttentionBlock\n\n\nclass NystromBlock(AttentionBlock):\n    pass\n"""
)
PY

echo "[refiner-megasam] installing Python dependencies"
python -m pip install --upgrade pip
if ! python - <<'PY'
import torch
PY
then
  python -m pip install \
    ${MEGASAM_TORCH_PACKAGES:-torch torchvision} \
    --index-url "${MEGASAM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
fi

python -m pip install wheel
python -m pip install \
  opencv-python-headless==4.9.0.80 \
  tqdm==4.67.1 \
  imageio==2.36.0 \
  imageio-ffmpeg==0.5.1 \
  einops==0.8.0 \
  scipy==1.14.1 \
  matplotlib==3.9.2 \
  timm==1.0.7 \
  ninja==1.11.1 \
  numpy==1.26.3 \
  "huggingface-hub>=0.25,<2" \
  kornia==0.7.4 \
  safetensors \
  pillow \
  click \
  trimesh \
  appdirs \
  fvcore==0.1.5.post20221221 \
  iopath \
  wandb \
  yacs

python -m pip install xformers==0.0.29.post3 --no-deps
TORCH_VERSION="$(python - <<'PY'
import torch
print(torch.__version__.split("+")[0])
PY
)"
CUDA_VERSION="$(python - <<'PY'
import torch
cuda = torch.version.cuda or ""
print("cu" + cuda.replace(".", ""))
PY
)"
python -m pip install torch-scatter==2.1.2 \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html"

python -m pip install --no-deps -e "$LINGBOT_DEPTH_ROOT"

# UniDepth is used via PYTHONPATH by refiner_export.py. Avoid installing the
# editable package because its full requirements pin torch and huggingface-hub
# versions that conflict with the Refiner cloud worker.

(
  cd "$MEGASAM_ROOT/base"
  echo "[refiner-megasam] building MegaSAM base extension"
  python - <<'PY'
from pathlib import Path

for path in [Path("setup.py")]:
    if not path.exists():
        continue
    text = path.read_text()
    if "compute_90" not in text:
        text = text.replace(
            "'nvcc': [",
            "'nvcc': ['-gencode=arch=compute_90,code=sm_90', ",
        )
    path.write_text(text)

for source in Path(".").rglob("*"):
    if source.suffix not in {".cpp", ".cu", ".h"}:
        continue
    text = source.read_text()
    text = text.replace(".type()", ".scalar_type()")
    text = text.replace(".device().scalar_type()", ".device().type()")
    source.write_text(text)
PY
  if ! python setup.py install > /tmp/megasam-base-build.log 2>&1; then
    tail -400 /tmp/megasam-base-build.log >&2
    exit 1
  fi
)

mkdir -p "$MEGASAM_ROOT/Depth-Anything/checkpoints" "$MEGASAM_ROOT/checkpoints"

download_if_missing() {
  local url="$1"
  local output="$2"
  if [ ! -s "$output" ]; then
    python - <<'PY' "$url" "$output"
from pathlib import Path
from urllib.request import urlretrieve
import sys

url, output = sys.argv[1], Path(sys.argv[2])
output.parent.mkdir(parents=True, exist_ok=True)
urlretrieve(url, output)
PY
  fi
}

download_if_missing \
  "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth" \
  "$MEGASAM_ROOT/Depth-Anything/checkpoints/depth_anything_vitl14.pth"

cp "$REFINER_EXPORT" "$MEGASAM_ROOT/refiner_export.py"

missing=0
for required in \
  "$MEGASAM_ROOT/checkpoints/megasam_final.pth" \
  "$MEGASAM_ROOT/Depth-Anything/checkpoints/depth_anything_vitl14.pth" \
  "$MEGASAM_ROOT/refiner_export.py"; do
  if [ ! -s "$required" ]; then
    echo "missing required MegaSAM runtime file: $required" >&2
    missing=1
  fi
done

if [ "$missing" -ne 0 ]; then
  exit 2
fi

echo "MegaSAM runtime installed at $MEGASAM_ROOT"
