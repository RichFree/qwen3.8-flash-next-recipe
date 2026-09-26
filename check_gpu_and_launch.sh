#!/usr/bin/env bash
# Pre-flight check, then exec into vLLM — runs as llama-swap's actual launched
# process for any model wired through the ${gpu_check} macro in
# llama-swap-config.yaml (currently qwen3.8-27b on GPU1, qwen3.8-flash-next
# on GPU0 — see that file's env: blocks for GPU_INDEX/UTILIZATION_PCT).
#
# Both GPUs on bfg are otherwise-shared (see README.md): GPU1 runs other
# people's transient jobs, GPU0 runs bimo's own always-resident deployment.
# qwen3.8-27b only spins down (ttl-idle) and back up because of this dynamic
# setup, which opens a window where another job (e.g. a training run) can
# claim the target GPU's now-free memory before the next request tries to
# reload us. If that happens, loading anyway means either an ugly vLLM
# CUDA-OOM crash mid weight-load or forcibly starving whatever's already
# running there. Refuse to start instead — the request fails fast and
# cleanly, and the GPU is left alone for whoever's using it.
#
# `exec`, not a subshell call — this script's PID becomes vLLM's PID, keeping
# the property config.yaml's comments rely on: llama-swap's SIGTERM (sent to
# only this one PID, no process-group kill) lands on vLLM itself.
set -euo pipefail

# Which physical GPU to check/launch on — 0 or 1, both RTX PRO 6000 Blackwell
# on bfg with identical VRAM, so TOTAL_MIB below doesn't need to vary by index.
# Set via the GPU_INDEX env var (llama-swap-config.yaml's ${gpu_index} macro
# feeds this through the model's env: block); defaults to 1 for anyone calling
# this script directly. GPU0 also runs bimo's own always-resident deployment
# (port 9132, unrelated to this repo) — switching here means racing that
# resident process for headroom instead of GPU1's transient shared jobs.
GPU_INDEX="${GPU_INDEX:-1}"
TOTAL_MIB=97887          # RTX PRO 6000 Blackwell on bfg — see nvidia-smi
# Must match the model's own --gpu-memory-utilization (as a whole-number
# percent, since bash arithmetic below is integer-only). Defaults to 85 to
# match qwen3.8-27b's existing behavior for anyone calling this script
# directly without setting it; llama-swap-config.yaml sets it explicitly
# per-model via env: (qwen3.8-27b: 85, qwen3.8-flash-next: 95).
UTILIZATION_PCT="${UTILIZATION_PCT:-85}"
REQUIRED_MIB=$(( TOTAL_MIB * UTILIZATION_PCT / 100 ))

VLLM_BIN="/home/richard/Projects/03_experiments/qwen3.8_hosting/.venv/bin/vllm"

# CUDA toolchain env — mirrors ~/.zshrc exactly (see that file). This is the
# actual fix for the FlashInfer JIT failures chased through most of
# 2026-09-23 (docs/gateway/guides/bfg-dynamic-cutover.md has the full
# investigation) — every run that worked (the original resident process,
# and our own first llama-swap cold start) was launched from an interactive
# shell that sources .zshrc; every run that failed was launched without it
# (systemd, or this script before this fix). Two earlier guesses were wrong:
# (1) plain absence of nvcc doesn't reliably disable FlashInfer — at least
# one op (flashinfer_xqa_batch_decode_with_kv_cache) has no non-FlashInfer
# fallback at all, so making has_flashinfer() False just makes it fail
# every time instead of intermittently; (2) putting the venv's own bundled
# nvcc (nvidia/cu13/bin, CUDA 13.3) on PATH is actively worse — it's a
# version mismatch against what PyTorch was built against (CUDA 13.0) and
# produces a hard "CUDA compiler and CUDA toolkit headers are incompatible"
# build error on any kernel not already cached. /usr/local/cuda (13.1) is
# the toolkit .zshrc actually points at, and matches closely enough to work.
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.1"

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_INDEX")
UTIL_PCT=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU_INDEX")

if (( FREE_MIB < REQUIRED_MIB )); then
    echo "REFUSING TO START: GPU${GPU_INDEX} has only ${FREE_MIB} MiB free" \
         "(need ~${REQUIRED_MIB} MiB for --gpu-memory-utilization 0.${UTILIZATION_PCT} of" \
         "${TOTAL_MIB} MiB total). Current utilization ${UTIL_PCT}% — looks like" \
         "another job (training run?) is using GPU${GPU_INDEX}." >&2
    exit 1
fi

exec "$VLLM_BIN" "$@"
