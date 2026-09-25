#!/usr/bin/env bash
set -euo pipefail

# SCAFFOLD — not validated. See models/qwen3.8-flash-next/README.md for what's
# unconfirmed (vLLM architecture support, GPU sizing, speculative-decode
# method name, vision/vocab flags) before running this for real.
if [ -z "${ALLOW_UNVALIDATED:-}" ]; then
  echo "models/qwen3.8-flash-next/serve.sh is an unvalidated scaffold." >&2
  echo "Read models/qwen3.8-flash-next/README.md first, then re-run with" \
       "ALLOW_UNVALIDATED=1 to proceed anyway." >&2
  exit 1
fi

cd "$(dirname "$0")/../.."
source .venv/bin/activate
source models/qwen3.8-flash-next/model.env

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TODO before trusting this:
# - Confirm this vLLM version (0.28.0 in .venv) actually supports the
#   "qwen4_exp" / Qwen4ExpForConditionalGeneration architecture (hybrid
#   linear+full attention MoE). The model card only links to an external
#   vLLM "recipe" page, no inline serve command — check that page.
# - --gpu-memory-utilization / --tensor-parallel-size: this checkpoint is
#   125B total / 6B active params (NVFP4) plus a large n-gram embedding
#   table and an MTP head — likely much tighter on a single ~98GB GPU than
#   the 27B model. Confirm it actually fits before assuming TP=1.
# - Built-in MTP speculative decoding: figure out the correct
#   --speculative-config "method" value for this checkpoint's MTP head
#   (nvfp4_experts_mtp.safetensors) — do not reuse "dflash", that's the
#   27B model's separate draft-model method and doesn't apply here.
# - --tool-call-parser / --reasoning-parser: copied qwen3-family guesses
#   below from the 27B model's flags; verify against this model's actual
#   chat template (chat_template.jinja in the checkpoint).
# - Vision: this checkpoint is vision-capable (image_token_id set,
#   preprocessor_config.json / video_preprocessor_config.json present).
#   May need --mm-encoder-tp-mode data like the 27B model's NVFP4 variant.
# - Long context: model card recommends YaRN rope overrides via
#   --hf-overrides for context > 262144 tokens — not applied here.

vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --port "$PORT" \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
  # --speculative-config: TODO — see notes above, method name unconfirmed.
