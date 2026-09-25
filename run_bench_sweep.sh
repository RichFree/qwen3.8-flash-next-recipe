#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

MODEL="${1:-qwen3.8-27b}"
MODEL_DIR="models/$MODEL"
source "$MODEL_DIR/model.env"

RESULT_DIR="$MODEL_DIR/bench_results"
mkdir -p "$RESULT_DIR"

DEPTHS=(1024 4096 16384 65536)
OUTPUT_LEN=256
NUM_PROMPTS=5
SEED=42

run_sweep () {
  local label=$1
  for depth in "${DEPTHS[@]}"; do
    echo "=== [$label] depth=$depth ==="
    vllm bench serve \
      --backend openai \
      --endpoint /v1/completions \
      --host 127.0.0.1 \
      --port "$PORT" \
      --model "$MODEL_NAME" \
      --tokenizer "$MODEL_PATH" \
      --dataset-name sonnet \
      --dataset-path ./sonnet.txt \
      --sonnet-input-len "$depth" \
      --sonnet-output-len "$OUTPUT_LEN" \
      --ignore-eos \
      --max-concurrency 1 \
      --num-prompts "$NUM_PROMPTS" \
      --num-warmups 1 \
      --seed "$SEED" \
      --percentile-metrics ttft,tpot,itl \
      --metric-percentiles 50,90,99 \
      --label "$label" \
      --save-result \
      --result-dir "$RESULT_DIR" \
      --result-filename "${label}_depth${depth}.json" \
      2>&1 | tee -a "$RESULT_DIR/${label}_depth${depth}.log"
  done
}

wait_for_ready () {
  local timeout_s=$1
  local waited=0
  until curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; do
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
  echo "Stopping server on port $PORT..."
  pkill -f "serve_nospec\.sh" 2>/dev/null || true
  pkill -f "serve\.sh" 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "VLLM::EngineCore" 2>/dev/null || true
  # wait for port to free
  local waited=0
  while (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; do
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
echo "# Phase 1: benchmarking SPEC-DECODE server  #"
echo "############################################"
wait_for_ready 30 || { echo "spec-decode server not ready, aborting"; exit 1; }
run_sweep "spec"

echo "############################################"
echo "# Phase 2: swap to NO-SPEC baseline server  #"
echo "############################################"
stop_server
nohup "./$MODEL_DIR/serve_nospec.sh" > "$MODEL_DIR/server_nospec.log" 2>&1 &
echo "nospec server PID: $!"
disown

# wait for the baseline server to finish loading (log-based, generous timeout)
echo "Waiting for nospec server startup (log-based)..."
waited=0
until grep -qE "Application startup complete|Traceback|address already in use" "$MODEL_DIR/server_nospec.log" 2>/dev/null; do
  sleep 2
  waited=$((waited + 2))
  if [ "$waited" -ge 300 ]; then
    echo "TIMEOUT waiting for nospec server startup log"
    break
  fi
done
tail -n 5 "$MODEL_DIR/server_nospec.log"
wait_for_ready 60 || { echo "nospec server not ready, aborting"; exit 1; }

echo "############################################"
echo "# Phase 3: benchmarking NO-SPEC server      #"
echo "############################################"
run_sweep "nospec"

echo "############################################"
echo "# Phase 4: restore SPEC-DECODE server       #"
echo "############################################"
stop_server
nohup "./$MODEL_DIR/serve.sh" > "$MODEL_DIR/server.log" 2>&1 &
echo "spec server PID: $!"
disown

echo "Waiting for spec server startup (log-based)..."
waited=0
until grep -qE "Application startup complete|Traceback|address already in use" "$MODEL_DIR/server.log" 2>/dev/null; do
  sleep 2
  waited=$((waited + 2))
  if [ "$waited" -ge 300 ]; then
    echo "TIMEOUT waiting for spec server restart log"
    break
  fi
done
tail -n 5 "$MODEL_DIR/server.log"
wait_for_ready 60 && echo "Spec-decode server restored and ready."

echo "ALL BENCHMARK PHASES COMPLETE"
