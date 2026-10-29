# qwen3.8_hosting

Always-on vLLM hosting for `qwen3.8-flash-next` (`Inferact/Qwen3.8-Flash-Next-NVFP4`)
on `bfg` (2x RTX PRO 6000 Blackwell, ~98GB each; both cards are shared with other
jobs). Model notes and the investigation behind the flags live in
`models/qwen3.8-flash-next/README.md`.

## Layout

```
models/qwen3.8-flash-next/
  model.env               MODEL_NAME, PORT (8000), MODEL_PATH.
  serve.sh                The launch: GPU free-memory check, CUDA env, exec vllm serve.
  bench_results/          vllm bench serve output (json + log per run).
  README.md               Validated config, what was tried and rejected, open items.
deploy/bfg-flash-next.service   systemd user unit template (`make install` renders it).
Makefile                  install/start/stop/status/logs/call/bench. See `make help`.
run_bench.sh              Depth / concurrency sweeps against the running server.
sonnet.txt                Benchmark dataset (vllm bench's "sonnet" format).
venv-snapshots/           pip freezes taken around the nightly-vLLM upgrade.
.venv/                    Python venv (gitignored).
```

## Setup (from scratch)

Needs: a GPU with ~93 GB free (RTX PRO 6000 Blackwell class), ~132 GB free host RAM,
CUDA toolkit at `/usr/local/cuda` (13.1 — `serve.sh` uses it for FlashInfer's JIT; the
venv's bundled nvcc breaks it), `uv`, and systemd user services.

1. **venv with nightly vLLM.** Stable 0.29/0.30 cannot offload the 95 GiB embedding table;
   only the nightly can.
   ```bash
   uv venv .venv --python 3.12
   uv pip install --python .venv/bin/python -U vllm --torch-backend=auto \
     --extra-index-url https://wheels.vllm.ai/nightly
   ```
   Validated against vLLM `0.30.1rc1.dev186` + torch `2.13.0+cu130`; the exact pins are in
   `venv-snapshots/requirements-2026-09-26-vllm0.30.1rc1-nightly.txt`. Installing from that
   file reproduces them more reliably than the moving nightly index.
2. **Weights.**
   ```bash
   hf download Inferact/Qwen3.8-Flash-Next-NVFP4
   ```
   Then set `MODEL_PATH` in `models/qwen3.8-flash-next/model.env` to the snapshot directory
   (it is hardcoded to `/data/huggingface/hub/...` for this box).
3. Check `PORT` in `model.env` and the GPU (`serve.sh` defaults to GPU0;
   `CUDA_VISIBLE_DEVICES=1` overrides), then smoke-test with `make serve-fg` before
   installing the service below.

## Running it

```bash
make install     # render deploy/bfg-flash-next.service into ~/.config/systemd/user
make start       # cold start takes ~3-4 minutes
make status      # service state, /health, served model
make logs
systemctl --user enable bfg-flash-next.service   # optional: start on login/boot
```

vLLM listens on its default `:8000`. It holds ~93 GB on one GPU and ~132 GB of host
RAM whenever it is up; there is no idle unload.

### Cutover from llama-swap

- **The gateway's tunnel still ends at `:12000`** (where llama-swap listened), not
  `:8000`. Until the tunnel (or a local forward) targets `:8000`, gateway requests for
  `qwen3.8-flash-next-bfg` will fail. That tunnel is set up outside this account.
- `bfg-llama-swap.service` must be stopped (and disabled): it also wants a whole GPU.
- Check `:8000` is free first (`ss -tlnp | grep :8000`); something else was bound to
  it when this was written.

## Notes

- **GPU choice is fixed, not dynamic.** `serve.sh` uses GPU0 (override with
  `Environment=CUDA_VISIBLE_DEVICES=1` in the unit, or in the shell for `serve-fg`).
  At `--gpu-memory-utilization 0.95` the card must be essentially empty; if it isn't,
  `serve.sh` exits 1 with `REFUSING TO START` rather than crash into another job, and
  the unit retries every 60 s.
- **Host RAM matters as much as VRAM**: `--cpu-offload-gb 100` + 32 GiB KV offload on a
  502 GB shared box.
- `make stop` stops the whole cgroup, which includes `VLLM::EngineCore`.
- `make serve-fg` is for debugging; stop the service first (same port, same GPU).
