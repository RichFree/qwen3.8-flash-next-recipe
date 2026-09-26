#!/usr/bin/env bash
set -euo pipefail

# Validated 2026-09-26 on GPU0 — see README.md's "2026-09-26: validated
# working config" section for the full investigation (why nightly vLLM, why
# these specific offload flags, why CUDA graphs needed --max-num-seqs
# dropped, why MTP speculative decoding was tried and rejected) and
# bench_results/ for the numbers. This is also the config running live under
# llama-swap-config.yaml's qwen3.8-flash-next entry; keep the two in sync.
if [ -z "${ALLOW_UNVALIDATED:-}" ]; then
  echo "models/qwen3.8-flash-next/serve.sh is validated but still marked" \
       "manual-only — read README.md, then re-run with ALLOW_UNVALIDATED=1." >&2
  exit 1
fi

cd "$(dirname "$0")/../.."
source .venv/bin/activate
source models/qwen3.8-flash-next/model.env

# GPU0, not GPU1 — GPU1 is bfg's shared GPU and (dynamically) hosts
# qwen3.8-27b via llama-swap. GPU0 is otherwise shared too (see README's GPU0
# caveat about bimo's own resident deployment sometimes living there) — check
# nvidia-smi -i 0 yourself before running this manually; unlike the
# llama-swap path, this script does not gate on GPU fit.
export CUDA_VISIBLE_DEVICES=0
# NOT expandable_segments:True (unlike qwen3.8-27b's env) — incompatible with
# --kv-transfer-config's OffloadingConnector (vLLM rejects it at startup).

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
