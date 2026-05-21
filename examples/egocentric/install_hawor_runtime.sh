#!/usr/bin/env bash
set -euo pipefail

HAWOR_ROOT="${HAWOR_ROOT:-/opt/HaWoR}"
HAWOR_REPO="${HAWOR_REPO:-https://github.com/ThunderVVV/HaWoR.git}"
HAWOR_REF="${HAWOR_REF:-main}"
REFINER_EXPORT="${REFINER_EXPORT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hawor_refiner_export.py}"
REFINER_PATCH_SAFETENSORS="${REFINER_PATCH_SAFETENSORS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patch_hawor_safetensors_native.py}"
HAWOR_SAFETENSORS_REPO="${HAWOR_SAFETENSORS_REPO:-macrodata/hawor-safetensors}"
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

if ! python - <<'PY'
import torch
PY
then
  python -m pip install \
    ${HAWOR_TORCH_PACKAGES:-torch torchvision} \
    --index-url "${HAWOR_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
fi

python -m pip install wheel ninja
python - <<'PY' "$HAWOR_ROOT/requirements.txt" /tmp/hawor-requirements-runtime.txt
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:])
skip = ("torch-scatter",)
lines = [
    line
    for line in source.read_text().splitlines()
    if line.strip() and not any(token in line for token in skip)
]
output.write_text("\n".join(lines) + "\n")
PY
python -m pip install --no-build-isolation -r /tmp/hawor-requirements-runtime.txt
TORCH_BASE_VERSION="$(python - <<'PY'
import torch
print(torch.__version__.split("+", 1)[0])
PY
)"
python -m pip install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_BASE_VERSION}+cu124.html"
python -m pip install pytorch-lightning==2.2.4 --no-deps
python -m pip install lightning-utilities torchmetrics==1.4.0
python -m pip install huggingface-hub safetensors scipy imageio-ffmpeg
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

(
  cd "$HAWOR_ROOT"
  python - <<'PY'
import pytorch3d
from lib.vis.renderer import Renderer

version = getattr(pytorch3d, "__version__", "unknown")
print(f"PyTorch3D renderer available: {version}; {Renderer.__name__}")
PY
)

(
  cd "$HAWOR_ROOT/thirdparty/DROID-SLAM"
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
  if ! python setup.py install > /tmp/droid-slam-build.log 2>&1; then
    tail -400 /tmp/droid-slam-build.log >&2
    exit 1
  fi
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
from urllib.request import urlretrieve
import sys

url, output = sys.argv[1], Path(sys.argv[2])
output.parent.mkdir(parents=True, exist_ok=True)
urlretrieve(url, output)
PY
  fi
}

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

python "$REFINER_PATCH_SAFETENSORS" \
  --repo-id "$HAWOR_SAFETENSORS_REPO" \
  --hawor-root "$HAWOR_ROOT"

cp "$REFINER_EXPORT" "$HAWOR_ROOT/refiner_export.py"

missing=0
for required in \
  "$HAWOR_ROOT/weights/safetensors/droid.safetensors" \
  "$HAWOR_ROOT/weights/safetensors/metric3d.safetensors" \
  "$HAWOR_ROOT/weights/safetensors/MANO_RIGHT.safetensors" \
  "$HAWOR_ROOT/weights/safetensors/MANO_LEFT.safetensors" \
  "$HAWOR_ROOT/refiner_safetensors_runtime.py" \
  "$HAWOR_ROOT/refiner_export.py"; do
  if [ ! -s "$required" ]; then
    echo "missing required HaWoR runtime file: $required" >&2
    missing=1
  fi
done

if [ "$missing" -ne 0 ]; then
  cat >&2 <<'EOF'
HaWoR runtime is partially installed, but required runtime files are missing.
The default setup patches HaWoR to read DROID, Metric3D, and MANO files from
safetensors in HAWOR_SAFETENSORS_REPO. Set HAWOR_SAFETENSORS_REPO to a readable
Hugging Face dataset repo.
EOF
  exit 2
fi

echo "HaWoR runtime installed at $HAWOR_ROOT"
