#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Baseline variant: identical to serve.sh but WITHOUT speculative decoding,
# for A/B benchmarking against the dflash-enabled server.
vllm serve /data/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/57926baca9a82b4d6906b43f2750d55315f5b10f \
  --served-model-name qwen3.8-27b \
  --port 12000 \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data
