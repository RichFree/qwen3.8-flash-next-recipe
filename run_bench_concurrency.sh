#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

RESULT_DIR=./bench_results
mkdir -p "$RESULT_DIR"

MODEL_PATH=/data/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/57926baca9a82b4d6906b43f2750d55315f5b10f

DEPTHS=(1024 4096 16384 65536)
CONCURRENCIES=(2 4)
OUTPUT_LEN=256
SEED=42

run_sweep () {
  local label=$1
  for conc in "${CONCURRENCIES[@]}"; do
    local num_prompts=$((conc * 2))
    for depth in "${DEPTHS[@]}"; do
      echo "=== [$label] concurrency=$conc depth=$depth num_prompts=$num_prompts ==="
      vllm bench serve \
        --backend openai \
        --endpoint /v1/completions \
        --host 127.0.0.1 \
        --port 12000 \
        --model qwen3.8-27b \
        --tokenizer "$MODEL_PATH" \
        --dataset-name sonnet \
        --dataset-path ./sonnet.txt \
        --sonnet-input-len "$depth" \
        --sonnet-output-len "$OUTPUT_LEN" \
        --ignore-eos \
        --max-concurrency "$conc" \
        --num-prompts "$num_prompts" \
        --num-warmups 1 \
        --seed "$SEED" \
        --percentile-metrics ttft,tpot,itl \
        --metric-percentiles 50,90,99 \
        --label "$label" \
        --save-result \
        --result-dir "$RESULT_DIR" \
        --result-filename "${label}_conc${conc}_depth${depth}.json" \
        2>&1 | tee -a "$RESULT_DIR/${label}_conc${conc}_depth${depth}.log"
    done
  done
}

wait_for_ready () {
  local timeout_s=$1
  local waited=0
  until curl -s -o /dev/null http://127.0.0.1:12000/health; do
    sleep 2
    waited=$((waited + 2))
    if [ "$waited" -ge "$timeout_s" ]; then
      echo "TIMEOUT waiting for server readiness after ${timeout_s}s"
      return 1
    fi
  done
  echo "Server ready after ${waited}s"
}

stop_server () {
  echo "Stopping server on port 12000..."
  pkill -f "serve_nospec.sh" 2>/dev/null || true
  pkill -f "\./serve\.sh" 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "VLLM::EngineCore" 2>/dev/null || true
  local waited=0
  while (echo > /dev/tcp/127.0.0.1/12000) 2>/dev/null; do
    sleep 2
    waited=$((waited + 2))
    if [ "$waited" -ge 60 ]; then
      echo "TIMEOUT waiting for port to free"
      break
    fi
  done
  sleep 3
}

echo "############################################"
echo "# Phase 1: concurrency sweep, SPEC server   #"
echo "############################################"
wait_for_ready 30 || { echo "spec-decode server not ready, aborting"; exit 1; }
run_sweep "spec"

echo "############################################"
echo "# Phase 2: swap to NO-SPEC baseline server  #"
echo "############################################"
stop_server
nohup ./serve_nospec.sh > server_nospec.log 2>&1 &
echo "nospec server PID: $!"
disown

echo "Waiting for nospec server startup (log-based)..."
waited=0
until grep -qE "Application startup complete|Traceback|address already in use" server_nospec.log 2>/dev/null; do
  sleep 2
  waited=$((waited + 2))
  if [ "$waited" -ge 300 ]; then
    echo "TIMEOUT waiting for nospec server startup log"
    break
  fi
done
tail -n 5 server_nospec.log
wait_for_ready 60 || { echo "nospec server not ready, aborting"; exit 1; }

echo "############################################"
echo "# Phase 3: concurrency sweep, NO-SPEC server #"
echo "############################################"
run_sweep "nospec"

echo "############################################"
echo "# Phase 4: restore SPEC-DECODE server       #"
echo "############################################"
stop_server
nohup ./serve.sh > server.log 2>&1 &
echo "spec server PID: $!"
disown

echo "Waiting for spec server startup (log-based)..."
waited=0
until grep -qE "Application startup complete|Traceback|address already in use" server.log 2>/dev/null; do
  sleep 2
  waited=$((waited + 2))
  if [ "$waited" -ge 300 ]; then
    echo "TIMEOUT waiting for spec server restart log"
    break
  fi
done
tail -n 5 server.log
wait_for_ready 60 && echo "Spec-decode server restored and ready."

echo "ALL CONCURRENCY BENCHMARK PHASES COMPLETE"
