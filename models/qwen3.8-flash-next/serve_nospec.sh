#!/usr/bin/env bash
set -euo pipefail

# Baseline variant — identical to serve.sh. There is no separate spec-decode
# variant for this model: MTP speculative decoding was tried and rejected
# (see README.md's "2026-09-26: validated working config" section — it made
# per-token latency worse on this hardware, despite a 46% acceptance rate).
# Kept as a separate script only for parity with the Makefile's serve/
# serve-nospec convention used by other models in this repo.
if [ -z "${ALLOW_UNVALIDATED:-}" ]; then
  echo "models/qwen3.8-flash-next/serve_nospec.sh is validated but still" \
       "marked manual-only — read README.md, then re-run with" \
       "ALLOW_UNVALIDATED=1." >&2
  exit 1
fi

cd "$(dirname "$0")/../.."
source .venv/bin/activate
source models/qwen3.8-flash-next/model.env

export CUDA_VISIBLE_DEVICES=0

vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --cpu-offload-gb 100 \
  --cpu-offload-params ngram_embedding \
  --offload-backend uva \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":34359738368,"blocks_per_chunk":4}}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
