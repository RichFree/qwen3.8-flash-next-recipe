# qwen3.8-flash-next (scaffold — not yet running)

Checkpoint: `Inferact/Qwen3.8-Flash-Next-NVFP4`
(`/data/huggingface/hub/models--Inferact--Qwen3.8-Flash-Next-NVFP4`)

This is **not** another quantization of the `qwen3.8-27b` model already hosted
in this repo — it's a different architecture entirely (`Qwen4ExpForConditionalGeneration`,
`model_type: qwen4_exp`): a preview of the Qwen4 line with hybrid linear
(Gated DeltaNet) + full (Qwen Sparse) attention, MoE, a built-in multi-token
prediction (MTP) head for speculative decoding, and vision input support.
125B total / 6B activated params, plus a 51B-entry n-gram embedding table —
roughly 4-5x the footprint of `qwen3.8-27b` on the same hardware.

`serve.sh` / `serve_nospec.sh` here are **scaffolds**: best-effort flags
carried over from the `qwen3.8-27b` scripts, refused-to-run by default
(`ALLOW_UNVALIDATED=1` required) until the items below are checked. Do not
wire this model into `llama-swap-config.yaml` for live/dynamic serving until
it has actually been validated manually.

## Open questions before running this for real

1. **vLLM architecture support.** Confirm the `.venv`'s vLLM (0.28.0 as of
   this scaffold) actually implements `qwen4_exp` / the hybrid attention +
   MTP head. The model card doesn't give an inline `vllm serve` command —
   it points at an external "recipe" page
   (`https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next`); read that first.
2. **GPU fit.** bfg's GPUs are ~98GB each (see `check_gpu_and_launch.sh`).
   This checkpoint is much bigger than `qwen3.8-27b`; a single GPU at
   `--tensor-parallel-size 1` may not fit it. Check actual VRAM usage before
   assuming single-GPU, and adjust `--gpu-memory-utilization` /
   `--tensor-parallel-size` accordingly. Note GPU1 is shared (see the
   `llama-swap-config.yaml` comments) and currently hosts `qwen3.8-27b`.
3. **Speculative decoding.** This model ships its own MTP head
   (`nvfp4_experts_mtp.safetensors`), not a separate draft model like
   `qwen3.8-27b`'s `dflash`. Find the correct `--speculative-config`
   `"method"` for vLLM's native MTP support for this checkpoint — don't
   reuse `"dflash"`, it doesn't apply here.
4. **Tool-call / reasoning parsers.** `--tool-call-parser qwen3_coder` and
   `--reasoning-parser qwen3` are copied from `qwen3.8-27b` as a starting
   guess. Verify against this checkpoint's own `chat_template.jinja`.
5. **Vision flags.** This checkpoint has `preprocessor_config.json` and
   `video_preprocessor_config.json` (image + video input). May need
   `--mm-encoder-tp-mode data` like `qwen3.8-27b`'s NVFP4 variant did.
6. **Long context / YaRN.** Native context is 262,144 tokens, extensible to
   1,000,000 via YaRN RoPE overrides (`--hf-overrides` with a `rope_type:
   yarn` block — see the model card's "Best Practices" section). Not
   applied in the scaffold; only add it if you actually need >262k context.

Once these are resolved and the model has been run and smoke-tested manually
(`make serve MODEL=qwen3.8-flash-next ALLOW_UNVALIDATED=1`, or
`./models/qwen3.8-flash-next/serve.sh` directly with the same env var), update
this README and consider adding a real (uncommented) entry to
`llama-swap-config.yaml`.
