#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "未找到 .venv，请先执行: ./setup.sh"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

export TORCH_HOME="$ROOT/.cache/torch"
export HF_HOME="$ROOT/.cache/huggingface"
export PYTHONUNBUFFERED=1
# 避免系统代理导致 Gradio 本地 startup-events 502
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"

PORT="${PORT:-7860}"
echo "启动全景人像消除 Demo → http://127.0.0.1:${PORT}"
echo "浏览器打开上述地址：上传全景 → 笔刷涂人 → 开始消除"
python -u app.py
