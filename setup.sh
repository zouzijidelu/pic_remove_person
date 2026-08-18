#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON_BIN:-}" && ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(printf '%s\n' "$PYTHON_BIN" | tail -n1)"
fi
if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
  :
elif [[ -x /opt/homebrew/bin/python3.12 ]]; then
  PYTHON_BIN="/opt/homebrew/bin/python3.12"
elif [[ -x /root/anaconda3/envs/pano-lama/bin/python ]]; then
  PYTHON_BIN="/root/anaconda3/envs/pano-lama/bin/python"
else
  PYTHON_BIN="$(command -v python3.12 || true)"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "需要 Python 3.12（Ubuntu 20.04 可用 conda create -n pano-lama python=3.12，再 PYTHON_BIN=... ./setup.sh）"
  exit 1
fi

echo "[setup] Python: $PYTHON_BIN ($($PYTHON_BIN --version))"

rm -rf .venv
"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
pip install -i "$PIP_INDEX" --upgrade pip

want_cuda="${TORCH_CUDA:-auto}"
if [[ "$want_cuda" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    want_cuda="cuda"
  else
    want_cuda="cpu"
  fi
fi

if [[ "$want_cuda" == "cuda" ]]; then
  TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
  echo "[setup] 安装 CUDA 版 PyTorch ← $TORCH_INDEX"
  pip install torch torchvision --index-url "$TORCH_INDEX"
else
  echo "[setup] 安装 CPU 版 PyTorch"
  pip install -i "$PIP_INDEX" torch torchvision
fi

pip install -i "$PIP_INDEX" \
  gradio simple-lama-inpainting opencv-python-headless segment-anything

if [[ "$want_cuda" == "cuda" ]]; then
  echo "[setup] 确保仍是 CUDA 版 torch（避免其它包把 CPU 轮子覆盖回来）"
  pip install torch torchvision --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
  python - <<'PY'
import torch
print(f"[setup] torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用：请检查驱动 / TORCH_INDEX（cu124 需驱动 ≥ 525）")
PY
fi

echo
echo "安装完成。运行: ./start.sh"
