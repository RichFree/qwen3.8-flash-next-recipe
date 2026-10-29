# syntax=docker/dockerfile:1
# Lean image for qwen3.8-flash-next. Multi-stage: the builder (CUDA devel image,
# uv, pip caches) is thrown away; only the venv is copied into a runtime image.
#
# Weights are NOT in the image — mount them (same path as model.env's default):
#   docker run --device nvidia.com/gpu=1 --ipc=host --ulimit memlock=-1:-1 \
#     -v /data/huggingface:/data/huggingface:ro -p 8000:8000 qwen3.8-flash-next
# --ipc=host and memlock matter: the 100 GB + 32 GiB host-RAM offloads use
# pinned memory.

ARG CUDA=13.0.2
ARG VLLM=0.32.0

FROM nvidia/cuda:${CUDA}-devel-ubuntu24.04 AS builder
ARG VLLM
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 \
    && rm -rf /var/lib/apt/lists/*
# Same distro + apt python as the runtime stage, so the venv's python symlink
# resolves there too. The venv path must be identical in both stages.
RUN uv venv /opt/venv --python /usr/bin/python3.12 \
    && uv pip install --python /opt/venv/bin/python --no-cache \
       "vllm==${VLLM}" --torch-backend=cu130

FROM nvidia/cuda:${CUDA}-runtime-ubuntu24.04
# FlashInfer JIT-compiles kernels on first use, so the runtime needs nvcc, the
# CUDA headers and a host compiler — but not the full devel image. The toolkit
# must stay at 13.0 to match the torch cu130 build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 g++ cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-cccl-13-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
COPY models/qwen3.8-flash-next/ /app/models/qwen3.8-flash-next/
ENV VLLM_BIN=/opt/venv/bin/vllm \
    PATH=/opt/venv/bin:$PATH \
    HF_HUB_OFFLINE=1
WORKDIR /app
EXPOSE 8000
ENTRYPOINT ["/app/models/qwen3.8-flash-next/serve.sh"]
