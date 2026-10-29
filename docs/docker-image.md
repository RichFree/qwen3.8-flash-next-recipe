# Building the Docker image

Status: **draft, written while the first build was still running.** Nothing here has been
verified beyond the Dockerfile parsing and the base-image tags existing. Fill in the
"Results" section once a build and a GPU 1 test run have finished.

## Goal

A lean image for `qwen3.8-flash-next` on stable vLLM (**0.32.0**, not the nightly the
host `.venv` uses), tested on GPU 1 so the live deployment on GPU 0 is untouched.

## How the image is built

`Dockerfile` is a two-stage build:

1. **builder** — `nvidia/cuda:13.0.2-devel-ubuntu24.04`. Installs `uv`, creates
   `/opt/venv` on the apt `python3.12`, and runs
   `uv pip install "vllm==0.32.0" --torch-backend=cu130 --no-cache`. This stage (compiler
   toolchain, uv, caches) is discarded.
2. **runtime** — `nvidia/cuda:13.0.2-runtime-ubuntu24.04`. Adds only what FlashInfer's
   first-use JIT compile needs (`g++`, `cuda-nvcc-13-0`, `cuda-cudart-dev-13-0`,
   `cuda-cccl-13-0`), copies `/opt/venv` from the builder, and copies `serve.sh` +
   `model.env` to `/app/models/qwen3.8-flash-next/`. Entrypoint is `serve.sh`.

Constraints that shaped it:

- **Venv path and python must match across stages.** Venvs hard-code their paths and
  symlink to the interpreter, so both stages use the same Ubuntu release, the apt
  `python3.12`, and `/opt/venv`.
- **CUDA toolkit stays at 13.0.x** to match the torch `cu130` build (vLLM 0.32.0 pins
  `torch==2.13.0`). Mixing in a newer nvcc caused a "toolkit headers are incompatible"
  failure on this repo before (see `serve.sh`).
- **A compiler is needed at runtime**, so the runtime base is not plain `-runtime`. The
  smaller alternative — precompiling FlashInfer kernels in the builder so the runtime needs
  no compiler — is untested.
- **Weights are not in the image.** They are mounted read-only.
- **`.dockerignore` is an allowlist** (`*`, then only `serve.sh` and `model.env`), so the
  build context stays tiny.
- `serve.sh` / `model.env` take overrides from the environment (`PORT`, `MODEL_PATH`,
  `VLLM_BIN`) so the same files work on the host and in the container.

## Build

```bash
docker build -t qwen3.8-flash-next:vllm0.32 .
# other versions:
docker build --build-arg VLLM=0.32.0 --build-arg CUDA=13.0.2 -t qwen3.8-flash-next:vllm0.32 .
docker images qwen3.8-flash-next:vllm0.32      # check the size
```

## Run (GPU 1 test)

GPUs are passed via CDI (`nvidia-ctk cdi list` shows `nvidia.com/gpu=0|1|all`; there is no
`nvidia` runtime in `docker info`, so `--gpus` is not the path used here).

```bash
docker run --rm --name flash-next-test \
  --device nvidia.com/gpu=1 \
  --ipc=host --ulimit memlock=-1:-1 \
  -v /data/huggingface:/data/huggingface:ro \
  -p 8002:8000 \
  qwen3.8-flash-next:vllm0.32
```

- Inside the container only GPU 1 is visible and appears as index 0, which is why
  `serve.sh`'s default `CUDA_VISIBLE_DEVICES=0` still works.
- Host ports 8000 and 8001 were already bound when this was written, so the test maps
  host `8002` to the container's `8000`.
- `--ipc=host` and `--ulimit memlock=-1:-1` are for the pinned host memory used by the
  100 GB embedding offload and 32 GiB KV offload. Needed amounts are unmeasured here.
- The test runs alongside the live deployment: it needs another ~132 GB of host RAM
  (about 311 GB was available) and a GPU with ~93 GB free (GPU 1 was at 657 MiB).

Smoke test: `curl localhost:8002/health`, then a chat completion to
`/v1/chat/completions` with `"model": "qwen3.8-flash-next"`.

## Things to watch on the first run

- **Does stable 0.32.0 offload the embedding table correctly?** Stable 0.29 and 0.30
  built the full 95 GiB `ngram_embedding` tensor on the GPU before offloading and OOMed;
  only the nightly worked (see `models/qwen3.8-flash-next/README.md`). 0.32.0's source has
  `qwen4_exp`, `OffloadingConnector` and the `uva` offloader, but whether it avoids that OOM
  is only answered by a cold start.
- Whether the flags from `serve.sh` are still accepted unchanged by 0.32.0.
- FlashInfer JIT compiling successfully with the minimal toolchain (missing header →
  add the matching `cuda-*-dev-13-0` package).
- `nvidia-smi` being available inside the container for `serve.sh`'s free-VRAM check.
- `VLLM_USE_V2_MODEL_RUNNER=1` (the `serve.sh` default) on 0.32.0 is untested.

## Results

_Not yet recorded: final image size, cold-start time, whether the offload worked, and any
Dockerfile changes the first run forced._
