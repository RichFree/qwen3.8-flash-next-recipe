#!/usr/bin/env bash
# Depth sweep (concurrency 1) and concurrency sweep (2, 4) against the
# already-running qwen3.8-flash-next server. Does not start or stop anything.
# Usage: ./run_bench.sh [depth|concurrency|all]   (default: all)
set -uo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate
source models/qwen3.8-flash-next/model.env

RESULT_DIR="models/qwen3.8-flash-next/bench_results"
mkdir -p "$RESULT_DIR"

DEPTHS=(1024 4096 16384 65536)
CONCURRENCIES=(2 4)
OUTPUT_LEN=256
SEED=42

curl -s -o /dev/null "http://127.0.0.1:$PORT/health" \
  || { echo "No server on port $PORT — start one first (make start)."; exit 1; }

bench () {
  local name=$1 conc=$2 depth=$3 num_prompts=$4
  echo "=== $name concurrency=$conc depth=$depth num_prompts=$num_prompts ==="
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
    --max-concurrency "$conc" \
    --num-prompts "$num_prompts" \
    --num-warmups 1 \
    --seed "$SEED" \
    --percentile-metrics ttft,tpot,itl \
    --metric-percentiles 50,90,99 \
    --label "$name" \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "${name}.json" \
    2>&1 | tee -a "$RESULT_DIR/${name}.log"
}

mode="${1:-all}"

if [[ $mode == depth || $mode == all ]]; then
  for depth in "${DEPTHS[@]}"; do
    bench "depth${depth}" 1 "$depth" 5
  done
fi

if [[ $mode == concurrency || $mode == all ]]; then
  for conc in "${CONCURRENCIES[@]}"; do
    for depth in "${DEPTHS[@]}"; do
      bench "conc${conc}_depth${depth}" "$conc" "$depth" $((conc * 2))
    done
  done
fi
