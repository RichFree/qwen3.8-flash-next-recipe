#!/usr/bin/env bash
# Serves qwen3.8-flash-next with vLLM. Run by deploy/bfg-flash-next.service
# (always-on), or by hand via `make serve-fg`.
#
# Validated 2026-09-26 — see README.md's "2026-09-26: validated working
# config" for why each flag is here (nightly vLLM, the two separate CPU
# offloads, --max-num-seqs instead of --max-model-len, no MTP) and
# bench_results/ for the numbers.
set -euo pipefail

cd "$(dirname "$0")/../.."
source models/qwen3.8-flash-next/model.env

# Which physical GPU. Both bfg cards are shared with other jobs; override with
# CUDA_VISIBLE_DEVICES=1 (or the unit's Environment=) to move the model.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TOTAL_MIB=97887          # RTX PRO 6000 Blackwell — see nvidia-smi
UTILIZATION_PCT=95       # must match --gpu-memory-utilization below
REQUIRED_MIB=$(( TOTAL_MIB * UTILIZATION_PCT / 100 ))

# CUDA toolchain env — mirrors ~/.zshrc. This is the actual fix for the
# FlashInfer JIT failures seen when vLLM was launched without an interactive
# shell (systemd): /usr/local/cuda (13.1) matches closely enough to what
# PyTorch was built against (13.0). The venv's bundled nvcc (cu13, 13.3) on
# PATH is worse — "CUDA compiler and CUDA toolkit headers are incompatible".
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.1"
# NOT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — vLLM rejects it
# together with the OffloadingConnector below.

# Refuse to start rather than load onto a card another job is using: loading
# anyway means a CUDA-OOM crash mid weight-load or starving that job.
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
if (( FREE_MIB < REQUIRED_MIB )); then
    echo "REFUSING TO START: GPU${CUDA_VISIBLE_DEVICES} has only ${FREE_MIB} MiB free" \
         "(need ~${REQUIRED_MIB} MiB for --gpu-memory-utilization 0.${UTILIZATION_PCT})." \
         "Another job is using it — see nvidia-smi." >&2
    exit 1
fi

# exec: this PID becomes vLLM's, so systemd's SIGTERM lands on vLLM itself.
exec .venv/bin/vllm serve "$MODEL_PATH" \
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
