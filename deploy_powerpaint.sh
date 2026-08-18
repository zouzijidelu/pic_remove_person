#!/usr/bin/env bash
# 在 GPU 服务器上为已部署的 LaMa Demo 加装 PowerPaint 旁路环境。
# 先把本机 iopaint-bench（不含 .venv）放到 /root/iopaint-bench，再执行：
#   bash /root/pano-inpaint/deploy_powerpaint.sh
set -euo pipefail

DEMO="$(cd "$(dirname "$0")" && pwd)"
BENCH="${IOPAINT_BENCH:-$(cd "$DEMO/.." && pwd)/iopaint-bench}"

if [[ ! -f "$BENCH/inpaint_worker.py" ]]; then
  echo "未找到 $BENCH/inpaint_worker.py"
  echo "请先把 iopaint-bench 解压到该目录（不要带上 Mac 的 .venv）"
  exit 1
fi

# iopaint 1.6.0 需要 Python 3.11（Pillow 9.5 无 3.12 wheel）。不要用 pano-lama 的 3.12。
if [[ -x /root/anaconda3/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/bin/python}"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.11)}"
elif [[ -x /root/anaconda3/envs/pano-lama/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/pano-lama/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

echo "[pp] bench=$BENCH"
echo "[pp] python=$PYTHON_BIN"

export PYTHON="$PYTHON_BIN"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
chmod +x "$BENCH/setup.sh"

venv_ok() {
  "$BENCH/.venv/bin/python" - <<'PY'
import iopaint, torch
assert torch.cuda.is_available()
print(f"[pp] 已有环境 torch={torch.__version__} cuda=True，跳过 setup")
PY
}

if [[ "${SKIP_SETUP:-0}" == "1" ]] || venv_ok; then
  :
else
  (cd "$BENCH" && ./setup.sh)
  if command -v nvidia-smi >/dev/null 2>&1; then
    if ! "$BENCH/.venv/bin/python" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
      echo "[pp] 安装 CUDA 版 torch，避免 iopaint 拉来 CPU 轮子"
      "$BENCH/.venv/bin/pip" install torch torchvision --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
    fi
    "$BENCH/.venv/bin/python" - <<'PY'
import torch
print(f"[pp] torch={torch.__version__} cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PowerPaint 环境 CUDA 不可用")
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
set_kv ENABLE_POWERPAINT 1
set_kv POWERPAINT_DEVICE cuda
set_kv POWERPAINT_MAX_SIDE 768
set_kv IOPAINT_BENCH "$BENCH"

echo "[pp] 已写入 $ENVF"
if systemctl is-enabled pano-lama >/dev/null 2>&1; then
  systemctl restart pano-lama
  echo "[pp] 已 restart pano-lama"
  echo "看日志: journalctl -u pano-lama -f"
  echo "界面出现 PowerPaint 后，第一次点击会加载约 4GB 权重（1～2 分钟），之后进程常驻。"
else
  echo "[pp] 未发现 systemd 服务，请手动 ./start.sh"
fi
