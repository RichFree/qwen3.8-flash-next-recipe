# qwen3.8_hosting

vLLM hosting scripts and benchmarks for Qwen3.8-family models on `bfg`
(2x RTX PRO 6000 Blackwell, ~98GB each; GPU1 is shared with other jobs).

## Layout

```
models/<name>/          One directory per model variant.
  model.env               MODEL_NAME, PORT, MODEL_PATH — sourced by serve.sh,
                           serve_nospec.sh, and the root bench scripts.
  serve.sh                Manual foreground/background launch, spec-decode.
  serve_nospec.sh          Same, without speculative decoding (A/B baseline).
  bench_results/           vllm bench serve output (json + log per run).
  bench_report.html        Rendered benchmark comparison, if generated.
  README.md                Model-specific notes, only where something is
                           non-obvious or unvalidated (see qwen3.8-flash-next/).

Makefile                 Generic targets, all take MODEL=<name>
                          (default: qwen3.8-27b). See `make help`.
run_bench_sweep.sh        Depth sweep (spec vs nospec), takes MODEL as $1.
run_bench_concurrency.sh  Concurrency sweep (spec vs nospec), takes MODEL as $1.
sonnet.txt                Shared benchmark dataset (vllm bench's "sonnet" format).
check_gpu_and_launch.sh   GPU-fit gate used by llama-swap-config.yaml (see below).
llama-swap-config.yaml    Dynamic-loading config for the llama-swap gateway.
llama-swap                Downloaded llama-swap binary (gitignored).
.venv/                    Shared Python venv for all models (gitignored).
```

## Two serving paths

1. **Manual** (`make serve` / `serve.sh`, port from each model's `model.env`):
   for local dev and benchmarking. Only one model runs at a time.
2. **llama-swap** (`llama-swap-config.yaml`, port 12000): a gateway that
   cold-starts a model's vLLM process on first request and stops it after an
   idle TTL, to free GPU1 for other jobs on the shared box. **This is
   typically the live path** — check `make status` or `ss -ltnp | grep 12000`
   before assuming nothing is running.

Currently only `qwen3.8-27b` has a live `llama-swap-config.yaml` entry.
`qwen3.8-flash-next` is scaffolded under `models/` but deliberately left
commented out there — see `models/qwen3.8-flash-next/README.md` for what's
still unvalidated before it should be wired in.

Note there's an existing, pre-existing discrepancy worth knowing about: the
manual `models/qwen3.8-27b/serve.sh` still points at the `unsloth` NVFP4
checkpoint, while `llama-swap-config.yaml`'s live entry was switched to a
different (FP8) checkpoint to work around a hardware-specific FlashInfer bug
(NVFP4 KV cache + masked multi-token verify on SM120) — see the comments in
`llama-swap-config.yaml`. The manual script was not updated to match; keep
that in mind if you use it for anything beyond quick local benchmarking.

## Adding a new model variant

1. `mkdir models/<name>` and add `model.env` (`MODEL_NAME`, `PORT`,
   `MODEL_PATH`), `serve.sh`, `serve_nospec.sh` (copy an existing model's as
   a starting point, then fix the flags for the new checkpoint's actual
   architecture/quantization).
2. `make serve MODEL=<name>` / `make bench MODEL=<name>` to exercise it
   manually before touching `llama-swap-config.yaml`.
3. Only once validated, add a real entry to `llama-swap-config.yaml`
   (there's a commented-out scaffold there for `qwen3.8-flash-next` showing
   the shape) and restart llama-swap to pick it up.

## Safety notes

- `make stop` (and the bench scripts' internal `stop_server`) kill **any**
  `vllm serve` / `VLLM::EngineCore` process by pattern match, not just the
  one this Makefile started. On this box that also matches llama-swap's own
  dynamically-launched backend. Don't run `make stop`/`make restart`/the
  bench scripts while a llama-swap-managed model is serving real traffic.
- `check_gpu_and_launch.sh` refuses to start a model if GPU1 doesn't have
  enough free memory (currently sized for `qwen3.8-27b`'s 85% utilization) —
  see that script's comments for why (shared-GPU cold-start race).
