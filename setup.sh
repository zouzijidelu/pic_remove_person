#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.12}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3.12 || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "需要 Python 3.12（当前系统的 3.13 与部分依赖不兼容）"
  exit 1
fi

rm -rf .venv
"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade pip
pip install -i https://mirrors.aliyun.com/pypi/simple \
  torch torchvision gradio simple-lama-inpainting opencv-python-headless

echo
echo "安装完成。运行: ./start.sh"
