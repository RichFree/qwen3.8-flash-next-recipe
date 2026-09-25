#!/usr/bin/env bash
set -euo pipefail

# SCAFFOLD — not validated. See models/qwen3.8-flash-next/README.md.
# Baseline variant: same as serve.sh minus speculative decoding (which is
# unconfigured there anyway pending the MTP method-name TODO).
if [ -z "${ALLOW_UNVALIDATED:-}" ]; then
  echo "models/qwen3.8-flash-next/serve_nospec.sh is an unvalidated scaffold." >&2
  echo "Read models/qwen3.8-flash-next/README.md first, then re-run with" \
       "ALLOW_UNVALIDATED=1 to proceed anyway." >&2
  exit 1
fi

cd "$(dirname "$0")/../.."
source .venv/bin/activate
source models/qwen3.8-flash-next/model.env

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --port "$PORT" \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
