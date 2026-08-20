#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

load_server_env() {
  local f="$1" line k v
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "${line// }" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    k="${line%%=*}"
    v="${line#*=}"
    [[ "$k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ ${#v} -ge 2 && ${v:0:1} == '"' && ${v: -1} == '"' ]]; then
      v="${v:1:${#v}-2}"
    elif [[ ${#v} -ge 2 && ${v:0:1} == "'" && ${v: -1} == "'" ]]; then
      v="${v:1:${#v}-2}"
    fi
    export "$k=$v"
  done < "$f"
}

if [[ -f "$ROOT/server.env" ]]; then
  load_server_env "$ROOT/server.env"
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
  echo "PowerPaint 未启用（ENABLE_POWERPAINT=0）"
else
  echo "PowerPaint 走旁路 iopaint-bench（需已 ./setup.sh）；设备可用 POWERPAINT_DEVICE"
fi
if [[ "${ENABLE_QWEN_EDIT:-auto}" == "0" ]]; then
  echo "Qwen-Edit 未启用（ENABLE_QWEN_EDIT=0）"
else
  echo "Qwen-Edit 走旁路 qwen-edit-bench；设备可用 QWEN_EDIT_DEVICE；默认不预热（WARMUP_QWEN=0）"
  if [[ "${QWEN_EDIT_DRY_RUN:-0}" == "1" ]]; then
    echo "Qwen-Edit 为 dry-run，不加载权重"
  fi
fi
exec "$PYTHON" -u app.py
