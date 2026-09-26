#!/usr/bin/env bash
# Picks a free GPU (by free-VRAM headroom, not just presence), then hands off
# to check_gpu_and_launch.sh for the final fit-check + exec into vllm. Runs
# as llama-swap's ${gpu_check} macro instead of that script directly, for any
# model that should float across GPUs rather than being pinned to one — see
# llama-swap-config.yaml.
#
# Why this exists: llama-swap itself has no concept of GPUs — it just runs
# whatever cmd:/env: a model config declares (confirmed 2026-09-26: it never
# inspects nvidia-smi or CUDA_VISIBLE_DEVICES itself). Every model previously
# had a fixed CUDA_VISIBLE_DEVICES baked into its config, which meant (a)
# manually editing YAML any time you wanted to move a model to the other GPU,
# and (b) a real collision risk if two models' fixed assignments landed on
# the same card (as they did on 2026-09-26, when qwen3.8-27b was temporarily
# hand-pinned to GPU0 to dodge a busy GPU1, landing on the same card as
# qwen3.8-flash-next's fixed GPU0 assignment — closable only by a human
# remembering not to trigger the other one). This script removes the need
# for any of that: every candidate GPU is checked at launch time, and
# whichever one actually has room wins, automatically and safely.
#
# `exec`, all the way through (this script -> check_gpu_and_launch.sh ->
# vllm) — same PID the whole way, which is what keeps the flock below held
# for the vllm process's entire lifetime without anything having to
# explicitly manage it.
set -euo pipefail

# Both bfg GPUs are candidates, tried in this order. No preference either
# way on purpose — first one with enough free memory wins. Override with
# GPU_CANDIDATES="1 0" etc. if that ever needs to change.
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"

# Must match the model's own --gpu-memory-utilization (whole-number percent —
# same convention as check_gpu_and_launch.sh, which reads the same env var
# again when this delegates to it). Required, not defaulted: a silently-wrong
# default here could pick a GPU that doesn't actually fit, since
# check_gpu_and_launch.sh's own recheck would use the same wrong default and
# agree with the mistake instead of catching it.
: "${UTILIZATION_PCT:?UTILIZATION_PCT must be set (see llama-swap-config.yaml env: block)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for GPU_INDEX in $GPU_CANDIDATES; do
    LOCK_FILE="/tmp/qwen3.8-hosting-gpu-${GPU_INDEX}.lock"

    # Non-blocking, fixed fd 200: if another launch already claimed this GPU
    # (still starting, or already running under this same mechanism), this
    # fails fast and we try the next candidate instead of queuing behind it.
    exec 200>"$LOCK_FILE"
    if ! flock -n 200; then
        exec 200>&-
        continue
    fi

    TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU_INDEX")
    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_INDEX")
    REQUIRED_MIB=$(( TOTAL_MIB * UTILIZATION_PCT / 100 ))

    if (( FREE_MIB >= REQUIRED_MIB )); then
        # Keep the lock (fd 200 stays open) — it's inherited across both the
        # exec below and the final exec into vllm inside
        # check_gpu_and_launch.sh (same PID the whole way), so it's held for
        # the vllm process's entire lifetime and released automatically by
        # the kernel when that process exits (ttl-idle stop, crash, or
        # otherwise) — nothing has to remember to clean it up. This is what
        # makes this safe against a second concurrent launch racing for the
        # same GPU: it fails the flock -n above, not just the free-memory
        # check, closing the check-then-act window a memory check alone
        # can't.
        export GPU_INDEX
        export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
        exec "$SCRIPT_DIR/check_gpu_and_launch.sh" "$@"
    fi

    # Not enough room on this one — release the lock and try the next
    # candidate. Someone else's job (not ours, so not lock-protected) could
    # be using it, same as check_gpu_and_launch.sh's own existing check.
    exec 200>&-
done

echo "REFUSING TO START: no candidate GPU (${GPU_CANDIDATES}) has enough free" \
     "memory for --gpu-memory-utilization 0.${UTILIZATION_PCT}. Run nvidia-smi" \
     "to see what's using them." >&2
exit 1
