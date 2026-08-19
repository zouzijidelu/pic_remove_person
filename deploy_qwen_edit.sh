#!/usr/bin/env bash
# 在 GPU 服务器上为已部署的 Demo 加装 Qwen-Image-Edit-2511 旁路。
# 先把本机 qwen-edit-bench（不含 .venv）放到 /root/qwen-edit-bench，再执行：
#   bash /root/pano-inpaint/deploy_qwen_edit.sh
# 不要装进 pano-inpaint 或 iopaint-bench 的 .venv。3090 不要 BF16。
set -euo pipefail

DEMO="$(cd "$(dirname "$0")" && pwd)"
BENCH="${QWEN_EDIT_BENCH:-$(cd "$DEMO/.." && pwd)/qwen-edit-bench}"

if [[ ! -f "$BENCH/edit_worker.py" ]]; then
  echo "未找到 $BENCH/edit_worker.py"
  echo "请先把 qwen-edit-bench 解压到该目录（不要带上 Mac 的 .venv）"
  exit 1
fi

if [[ -x /root/anaconda3/envs/pano-lama/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/pano-lama/bin/python}"
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.12)}"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.11)}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

echo "[qwen] bench=$BENCH"
echo "[qwen] python=$PYTHON_BIN"

export PYTHON="$PYTHON_BIN"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
chmod +x "$BENCH/setup.sh"

venv_ok() {
  "$BENCH/.venv/bin/python" - <<'PY'
import torch
from diffusers import QwenImageEditPlusPipeline  # noqa: F401
assert torch.cuda.is_available()
print(f"[qwen] 已有环境 torch={torch.__version__} cuda=True，跳过 setup")
PY
}

if [[ "${SKIP_SETUP:-0}" == "1" ]] || venv_ok; then
  :
else
  (cd "$BENCH" && ./setup.sh)
  if command -v nvidia-smi >/dev/null 2>&1; then
    if ! "$BENCH/.venv/bin/python" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
      echo "[qwen] 安装 CUDA 版 torch"
      "$BENCH/.venv/bin/pip" install torch torchvision --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
    fi
    "$BENCH/.venv/bin/python" - <<'PY'
import torch
print(f"[qwen] torch={torch.__version__} cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("Qwen-Edit 环境 CUDA 不可用")
PY
  fi
fi

ENVF="$DEMO/server.env"
if [[ ! -f "$ENVF" ]]; then
  cp "$DEMO/server.env.example" "$ENVF"
fi
touch "$ENVF"
set_kv() {
  local k="$1" v="$2"
  if grep -q "^${k}=" "$ENVF"; then
    sed -i "s|^${k}=.*|${k}=${v}|" "$ENVF"
  else
    echo "${k}=${v}" >> "$ENVF"
  fi
}
set_kv ENABLE_QWEN_EDIT 1
set_kv QWEN_EDIT_DEVICE cuda
set_kv QWEN_EDIT_MAX_SIDE 768
set_kv QWEN_EDIT_MARGIN 256
set_kv QWEN_EDIT_BENCH "$BENCH"
set_kv QWEN_EDIT_DRY_RUN 0
set_kv WARMUP_QWEN 0

echo "[qwen] 已写入 $ENVF"
echo "[qwen] 第一次点 Qwen-Edit 会下载量化权重（约 15GB+），请确认磁盘充足。"
echo "[qwen] 不要与 PowerPaint 同时 warmup 常驻；两者会互斥释放显存。"
if systemctl is-enabled pano-lama >/dev/null 2>&1; then
  systemctl restart pano-lama
  echo "[qwen] 已 restart pano-lama"
  echo "看日志: journalctl -u pano-lama -f"
else
  echo "[qwen] 未发现 systemd 服务，请手动 ./start.sh"
fi
