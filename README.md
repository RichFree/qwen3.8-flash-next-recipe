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

### Gateway tunnel

- **The gateway's tunnel must target `:8000`** (vLLM's port). If it still lands on a
  different local port, gateway requests for `qwen3.8-flash-next-bfg` will fail. That
  tunnel is set up outside this account.
- Check `:8000` is free first (`ss -tlnp | grep :8000`); something else was bound to
  it when this was written.

## Docker image

The same server can run from a container built on stable vLLM 0.32.0 (weights are mounted,
not baked in):

```bash
make docker-build
make docker-run [DOCKER_GPU=1] [DOCKER_PORT=8002]
```

See `docs/docker-image.md` for how the image is built, the toolchain packages it needs, the
JIT cache mount, and the results of the first GPU run.

## Configuration

### Model path

`serve.sh` loads `MODEL_PATH` from `models/qwen3.8-flash-next/model.env`, which only supplies
a default (`${MODEL_PATH:-...}`), so an environment variable wins. It must point at a
**local snapshot directory** (the one containing `config.json` and the safetensors), e.g. the
output of `hf download Inferact/Qwen3.8-Flash-Next-NVFP4`:

- **Permanently:** edit the default in `model.env`.
- **One run (host):** `MODEL_PATH=/path/to/snapshot make serve-fg`
- **systemd unit:** add `Environment=MODEL_PATH=/path/to/snapshot` to
  `deploy/bfg-flash-next.service` (then `make install && make restart`).
- **Docker:** `make docker-run DOCKER_MODELS=/my/models MODEL_PATH=/my/models/qwen3.8-flash-next`.
  `DOCKER_MODELS` is mounted read-only at the same path in the container (default
  `/data/huggingface`), and `MODEL_PATH` (which must be under it) is passed through as `-e`.
  The image sets `HF_HUB_OFFLINE=1`, so a Hugging Face repo id will not be downloaded
  at start; use a local path.

### Options

Environment variables read by `serve.sh` / `model.env` (set them in the shell, in the unit
with `Environment=`, or with `docker run -e`):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | snapshot under `/data/huggingface/hub/...` | Weights directory (see above). |
| `PORT` | `8000` | Port vLLM listens on. In Docker leave it at 8000 and remap with `-p HOST:8000`. |
| `CUDA_VISIBLE_DEVICES` | `0` | Which physical GPU to use. `serve.sh` also checks that GPU for free memory. |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | Set `0` to use vLLM's V1 model runner. |
| `VLLM_BIN` | `.venv/bin/vllm` | vLLM executable (the image sets `/opt/venv/bin/vllm`). |

Make variables:

| Variable | Default | Meaning |
|---|---|---|
| `DOCKER_GPU` | `1` | GPU index passed into the container (it sees it as GPU 0). |
| `DOCKER_PORT` | `8002` | Host port mapped to the container's `8000`. |
| `DOCKER_MODELS` | `/data/huggingface` | Host dir with the weights, mounted read-only at the same path. |
| `DOCKER_CACHE` | `~/.cache/flash-next-docker` | Host dir for FlashInfer/triton/vLLM JIT caches. |
| `IMAGE` | `qwen3.8-flash-next:vllm0.32` | Image tag built and run. |

`MODEL_NAME` (`qwen3.8-flash-next`, the served model name clients request) is set in
`model.env` as a fixed value, not an overridable default.

**Fixed in `serve.sh`** — edit the script to change these (there is no argument
passthrough). The values were validated together; see `models/qwen3.8-flash-next/README.md`
before changing them:

- `--gpu-memory-utilization 0.95`. If you change it, also change `UTILIZATION_PCT`
  (free-memory check), and `TOTAL_MIB` if the GPU is not a 97887 MiB card.
- `--max-model-len 262144`, `--max-num-seqs 8`
- `--cpu-offload-gb 100` with `--cpu-offload-params ngram_embedding` (uva backend) and a
  32 GiB (`cpu_bytes_to_use`) KV offload. These set the host-RAM requirement.
- `--tensor-parallel-size 1`
- Tool calling (`qwen3_coder` parser) and reasoning (`qwen3` parser)

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
