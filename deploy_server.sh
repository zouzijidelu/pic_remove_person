#!/usr/bin/env bash
# 在 Ubuntu GPU 服务器上执行（root 或有 sudo）。仅部署 LaMa。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

echo "[deploy] 目录: $ROOT"
if [[ "${SKIP_APT:-0}" != "1" ]]; then
  if ! $SUDO apt-get update -y; then
    echo "[deploy] apt-get update 失败（常见于无关的 NVIDIA container 源 404），忽略后继续"
  fi
  if ! $SUDO apt-get install -y python3-venv build-essential wget libglib2.0-0 libgl1-mesa-glx \
    && ! $SUDO apt-get install -y python3-venv build-essential wget libglib2.0-0 libgl1; then
    echo "[deploy] 系统包安装失败或已存在，若后面 conda/Python 可用则可忽略"
  fi
else
  echo "[deploy] SKIP_APT=1，跳过 apt"
fi

conda_python312() {
  # 日志走 stderr，stdout 只输出解释器路径，避免污染 PYTHON_BIN
  if ! conda env list 2>/dev/null | awk '{print $1}' | grep -qx 'pano-lama'; then
    conda create -y -n pano-lama python=3.12 pip >&2
  fi
  conda run -n pano-lama python -c 'import sys; print(sys.executable)'
}

pick_python312() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return
  fi
  if command -v python3.12 >/dev/null 2>&1; then
    command -v python3.12
    return
  fi
  local conda_sh="" conda_base=""
  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$conda_base/etc/profile.d/conda.sh"
      conda_python312
      return
    fi
  fi
  for conda_sh in \
    /root/miniconda3/etc/profile.d/conda.sh \
    /root/anaconda3/etc/profile.d/conda.sh \
    /opt/conda/etc/profile.d/conda.sh \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"
  do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck disable=SC1090
      source "$conda_sh"
      conda_python312
      return
    fi
  done
  echo "[deploy] 未找到 Python 3.12，尝试 deadsnakes PPA" >&2
  $SUDO apt-get install -y software-properties-common
  $SUDO add-apt-repository -y ppa:deadsnakes/ppa
  $SUDO apt-get update -y
  $SUDO apt-get install -y python3.12 python3.12-venv python3.12-dev
  command -v python3.12
}

PYTHON_BIN="$(pick_python312)"
export PYTHON_BIN
echo "[deploy] PYTHON_BIN=$PYTHON_BIN"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[deploy] 未检测到 nvidia-smi，将安装 CPU 版 torch（LaMa 会很慢）" >&2
  export TORCH_CUDA=cpu
else
  nvidia-smi -L || true
  export TORCH_CUDA=cuda
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
chmod +x setup.sh start.sh
./setup.sh

cp -n server.env.example server.env
# 3090 24GB：LaMa 可把推理边长开到 2048
if grep -q '^MAX_INFER_SIDE=' server.env; then
  sed -i 's/^MAX_INFER_SIDE=.*/MAX_INFER_SIDE=2048/' server.env
fi

UNIT=/etc/systemd/system/pano-lama.service
echo "[deploy] 写入 $UNIT"
$SUDO tee "$UNIT" >/dev/null <<EOF
[Unit]
Description=Panorama person inpaint (LaMa)
After=network.target

[Service]
Type=simple
WorkingDirectory=$ROOT
EnvironmentFile=-$ROOT/server.env
Environment=HF_ENDPOINT=https://hf-mirror.com
ExecStart=$ROOT/start.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

$SUDO systemctl daemon-reload
$SUDO systemctl enable --now pano-lama
sleep 2
$SUDO systemctl --no-pager --full status pano-lama || true

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
PORT="$(awk -F= '/^PORT=/{print $2}' server.env 2>/dev/null || echo 7860)"
echo
echo "已启动。内网浏览器打开: http://${IP:-<本机IP>}:${PORT:-7860}"
echo "看日志: journalctl -u pano-lama -f"
echo "首次会下载 LaMa（约 196MB）和 SAM（约 375MB），请等日志出现 Running on"
