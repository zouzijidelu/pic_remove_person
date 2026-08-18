#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/server.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/server.env"
  set +a
fi

if [[ ! -d .venv ]]; then
  echo "未找到 .venv，请先执行: ./setup.sh"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
PYTHON="$ROOT/.venv/bin/python"

export TORCH_HOME="$ROOT/.cache/torch"
export HF_HOME="$ROOT/.cache/huggingface"
export PYTHONUNBUFFERED=1
# 避免系统代理导致 Gradio 本地 startup-events 502
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7860}"
export HOST PORT
echo "启动全景人像消除 Demo → http://${HOST}:${PORT}"
echo "浏览器打开上述地址：上传全景 → 点选人像 → 选修复模型 → 开始消除"
if [[ "${ENABLE_POWERPAINT:-auto}" == "0" ]]; then
  echo "本次仅启用 LaMa（ENABLE_POWERPAINT=0）"
else
  echo "PowerPaint 走旁路 iopaint-bench（需已 ./setup.sh）；设备可用 POWERPAINT_DEVICE"
fi
exec "$PYTHON" -u app.py
