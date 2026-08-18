#!/usr/bin/env bash
# 在本机执行，生成可经 JumpServer「文件传输」上传的压缩包（不含 .venv）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$ROOT/../pano-lama-deploy.tar.gz}"

tar -C "$ROOT" -czf "$OUT" \
  --exclude '.venv' \
  --exclude '.cache' \
  --exclude 'output' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'pano-lama-deploy.tar.gz' \
  .

echo "已打包: $OUT"
echo "上传到 GPU 机后解压，再执行: bash deploy_server.sh"
echo "可选：把本机 .cache/sam/sam_vit_b_01ec64.pth 一并传上去，能跳过 SAM 权重下载"
