# Building and running the Docker image

A lean image for `qwen3.8-flash-next` on **stable vLLM 0.32.0** (not the nightly the host
`.venv` uses). Weights are not in the image; they are mounted. First built and run
2026-10-29 — the results and the problems that run exposed are recorded below.

## TL;DR

```bash
make docker-build                       # ~4-6 min once the builder layers are cached
make docker-run                         # GPU 1 by default; first start also JIT-compiles kernels
make docker-run DOCKER_GPU=0 DOCKER_PORT=8002   # what the first test actually used
curl localhost:8002/health
```

## Prerequisites

- Docker with GPUs available through **CDI**: `nvidia-ctk cdi list` must show
  `nvidia.com/gpu=0|1|all`. This box has no `nvidia` runtime in `docker info`, so
  `--gpus` is not the mechanism; `--device nvidia.com/gpu=N` is.
- The weights on the host at `/data/huggingface/hub/models--Inferact--Qwen3.8-Flash-Next-NVFP4/...`
  (the path baked into `model.env`; override with `-e MODEL_PATH=...`).
- A GPU with ~93 GB free (`--gpu-memory-utilization 0.95`) and ~132 GB of free host RAM for
  the CPU offloads.
- ~20 GB of disk for the image.

## How the image is built

`Dockerfile` is a two-stage build (`docker build -t qwen3.8-flash-next:vllm0.32 .`, wrapped
by `make docker-build`; override `IMAGE=`).

### Stage 1 — `builder` (thrown away)

`nvidia/cuda:13.0.2-devel-ubuntu24.04`, plus `uv` and apt `python3.12`:

```dockerfile
RUN uv venv /opt/venv --python /usr/bin/python3.12 \
    && uv pip install --python /opt/venv/bin/python --no-cache \
       "vllm==${VLLM}" --torch-backend=cu130
```

`VLLM=0.32.0` is a build arg (`--build-arg VLLM=...`); 0.32.0 pins `torch==2.13.0` and
`flashinfer-python==0.7.0.post1`. `--torch-backend=cu130` selects the CUDA 13.0 torch build.
Nothing from this stage is kept except `/opt/venv`.

### Stage 2 — runtime

`nvidia/cuda:13.0.2-runtime-ubuntu24.04` plus the packages below, then
`COPY --from=builder /opt/venv /opt/venv` and `serve.sh` + `model.env` into
`/app/models/qwen3.8-flash-next/`. The entrypoint is `serve.sh`.

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev g++ cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-cccl-13-0 \
        libcublas-dev-13-0=13.1.0.3-1 libcurand-dev-13-0 cuda-nvrtc-dev-13-0 \
    && rm -rf /var/lib/apt/lists/*
```

Why a runtime stage needs a compiler at all: FlashInfer JIT-compiles CUDA kernels the first
time a configuration is used, and triton builds a small C helper against `Python.h` at
import. So "lean" here means *runtime base + a minimal toolchain*, not a bare runtime image.

### Constraints that shaped it

- **Same distro, same python, same path in both stages.** A venv hard-codes its own path and
  symlinks to the interpreter, so both stages use Ubuntu 24.04, the apt `python3.12`, and
  `/opt/venv`.
- **CUDA toolkit stays at 13.0.x** to match torch's `cu130` build. A newer `nvcc` caused
  "toolkit headers are incompatible" failures earlier in this repo (see `serve.sh`).
- **`.dockerignore` is an allowlist** (`*`, then only `serve.sh` and `model.env`), so the
  build context is a few KB.
- `serve.sh` / `model.env` read `PORT`, `MODEL_PATH` and `VLLM_BIN` from the environment so the
  same files serve both the host and the container. `serve.sh` defaults to
  `CUDA_VISIBLE_DEVICES=0`, which is correct in the container: only the one GPU passed in is
  visible, as index 0.

### Why each toolchain package is there (found by failed runs, in this order)

| Failure on first start | Fix |
|---|---|
| `Python.h: No such file or directory` while inspecting the model architecture (triton's C helper) | `python3.12-dev` |
| FlashInfer `fused_moe_120` JIT: `cublasLt.h`, `curand_kernel.h`, `nvrtc.h` not found | `libcublas-dev-13-0`, `libcurand-dev-13-0`, `cuda-nvrtc-dev-13-0` |
| `libcublas-dev-13-0` wanted `libcublas-13-0 >= 13.1.1.3`, but the base image holds it at 13.1.0.3 | pin `libcublas-dev-13-0=13.1.0.3-1` |

Expect more of this if the model or vLLM version changes: a new JIT kernel can `#include` a
header from another CUDA library, and the symptom is a `fatal error: <header>: No such file`
in the container log. Find the package with `apt-cache search` / `apt-file`, and match the
version the base image already pins.

Image size by iteration: 17.5 GB (first build) → 17.6 GB (+ python headers) → **19.6 GB**
(+ the cuBLAS/cuRAND/NVRTC dev packages).

## Running it

```bash
make docker-run [DOCKER_GPU=1] [DOCKER_PORT=8002] [DOCKER_CACHE=~/.cache/flash-next-docker]
```

which expands to:

```bash
docker run --rm --name flash-next-gpu$DOCKER_GPU \
  --user $(id -u):$(id -g) \
  --device nvidia.com/gpu=$DOCKER_GPU --ipc=host --ulimit memlock=-1:-1 \
  -v /data/huggingface:/data/huggingface:ro \
  -v $DOCKER_CACHE:/cache \
  -e HOME=/cache -e VLLM_CACHE_ROOT=/cache/vllm -e TRITON_CACHE_DIR=/cache/triton \
  -p $DOCKER_PORT:8000 qwen3.8-flash-next:vllm0.32
```

- `--device nvidia.com/gpu=N`: CDI GPU passthrough.
- `--ipc=host --ulimit memlock=-1:-1`: the 100 GB embedding offload and 32 GiB KV offload use
  pinned host memory. (Started fine with these; whether both are strictly required is untested.)
- `-v /data/huggingface:...:ro`: weights, same path as in `model.env`.
- `-p`: host port → the container's `8000` (vLLM's default). Host `8000` and `8001` were
  already bound on this box, hence `8002`.
- `--user`: run as the host user so nothing in the cache ends up root-owned.

### The cache bind mount

JIT output must outlive the container, or every `--rm` start recompiles everything. This is a
**bind mount of a host directory** (default `~/.cache/flash-next-docker` → `/cache`), not a named
Docker volume; the compile (`nvcc`, `g++`, `ninja`) runs inside the container and writes into it.
`HOME=/cache` makes FlashInfer write to `/cache/.cache/flashinfer/<ver>/<arch>/`.

After the first successful start it held about 178 MB:

| Path under the cache dir | Size | What |
|---|---|---|
| `.cache/flashinfer/0.7.0.post1/120f/` | 152 MB | JIT-compiled FlashInfer kernels (incl. `fused_moe_120`) |
| `triton/` | 21 MB | triton's compiled helper |
| `vllm/` | 16 KB | almost empty — vLLM's `torch_compile_cache` did not land here; unexplained |

The FlashInfer cache directory is keyed by FlashInfer version (`0.7.0.post1` here), so the
host's existing `~/.cache/flashinfer/0.7.0` is not reused.

## Results of the first run (2026-10-29)

Run on **GPU 0**, not GPU 1: GPU 1 had a transient ~29 GB job on it at the time, and GPU 0 was
freed deliberately for the test.

- **Started and served.** `/health` → 200; a chat completion returned `pong` with
  `system_fingerprint: vllm-0.32.0-…` — i.e. stable 0.32.0, not the nightly.
- **The embedding-table offload worked on stable.** Stable 0.29/0.30 OOMed on the 95 GiB
  `ngram_embedding` table; 0.32.0 logs `EngramConfig(cpu_offload=True, ...)`, loaded the weights
  in **59.8 s**, and came up without an OOM. This was the main unknown going in.
- **Cold start: 463 s** to `Application startup complete`. This is *not* a fully cold number:
  the run before it had already compiled part of the FlashInfer cache before it crashed, so some
  kernels were reused. A truly empty-cache start would be longer.
- **Not measured:** the warm (second) start with the full cache, decode speed against
  `models/qwen3.8-flash-next/bench_results/`, tool calling, and vision.
- `serve.sh` enables `VLLM_USE_V2_MODEL_RUNNER=1` by default, so this run used it. Its effect
  is still unbenchmarked.
- `serve.sh`'s free-VRAM check refused to start twice while another job briefly held GPU memory
  (once on physical GPU 1 with 68861 MiB free, once on GPU 0 with 92871 MiB free; both need
  ~92992 MiB — the log says "GPU0" either way, since the container sees its one GPU as index 0)
  and passed on retry. That is the check working as designed on a shared box.

## Open items

- Time a warm second start to quantify what the cache mount buys.
- Try FlashInfer's prebuilt kernel wheel (`flashinfer-jit-cache`) to avoid compiling at all; unknown
  whether it covers this model on SM120. If it does, the runtime stage might drop `nvcc`.
- Find out why `vllm/` in the cache stays empty.
- Benchmark against the nightly numbers (`run_bench.sh` against `localhost:$DOCKER_PORT`).
- Decide whether the container replaces `deploy/bfg-flash-next.service`; today they are
  independent paths.
