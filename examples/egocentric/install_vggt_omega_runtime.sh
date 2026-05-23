#!/usr/bin/env bash
set -euo pipefail

INSTALL_ROOT="${INSTALL_ROOT:-/opt}"
REPO_DIR="${VGGT_OMEGA_REPO:-${INSTALL_ROOT}/vggt-omega}"
REPO_URL="${VGGT_OMEGA_REPO_URL:-https://github.com/facebookresearch/vggt-omega.git}"

if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -U "huggingface_hub[hf_xet]" hf_transfer imageio-ffmpeg requests

echo "VGGT-Omega runtime installed at ${REPO_DIR}"
echo "Download an approved checkpoint from https://huggingface.co/facebook/VGGT-Omega"
