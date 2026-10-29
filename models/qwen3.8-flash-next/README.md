# qwen3.8-flash-next

Checkpoint: `Inferact/Qwen3.8-Flash-Next-NVFP4`
(`/data/huggingface/hub/models--Inferact--Qwen3.8-Flash-Next-NVFP4`)

This is **not** another quantization of the `qwen3.8-27b` model (previously hosted
in this repo, removed on the `flash-next-only` branch) — it's a different architecture entirely (`Qwen4ExpForConditionalGeneration`,
`model_type: qwen4_exp`): a preview of the Qwen4 line with hybrid linear
(Gated DeltaNet) + full (Qwen Sparse) attention, MoE, a built-in multi-token
prediction (MTP) head for speculative decoding, and vision input support.
125B total / 6B activated params, plus a 95.37GiB n-gram "PLE" embedding
table — roughly 4-5x the footprint of `qwen3.8-27b` on the same hardware.

**Status: validated and live.** Runs always-on as `bfg-flash-next.service`
(`make install && make start`), launched by `serve.sh` on port 8000.
Tool-call/reasoning parsers are still unverified (see "Still open" below).

## 2026-09-26: validated working config

Getting this running on a single ~98GB GPU took several stacked fixes, each
forced by a real, measured blocker — not guesswork. In order:

1. **Nightly vLLM required.** `qwen4_exp` architecture support landed after
   0.28.0 (this repo's pinned version at the time). Stable 0.29.0 and 0.30.0
   both implement the architecture but **cannot** keep the PLE table off
   GPU (see point 2) — only the nightly build's `OffloadingConnector`/UVA
   offload path actually works. Installed via:
   ```
   uv pip install --python .venv/bin/python -U vllm --torch-backend=auto \
     --extra-index-url https://wheels.vllm.ai/nightly
   ```
   Resolved to `vllm==0.30.1rc1.dev186+g9bf44c401`. `torch` stayed at
   `2.13.0+cu130` throughout both the 0.28.0→0.29.0 and 0.29.0→nightly
   upgrades — only `vllm`/`flashinfer-python` (and CUDA-toolchain
   sub-packages) moved. Pre-upgrade pins are in `../../venv-snapshots/`,
   and a full `.venv` backup is at
   `../../.venv-backup-vllm0.28.0-20260926/` (gitignored, local-only) if
   `qwen3.8-27b` ever needs rolling back.

2. **The 95.37GiB PLE `ngram_embedding` table must be offloaded to host
   RAM.** Confirmed via `model.safetensors.index.json` + safetensors
   headers: `model.language_model.layers.1.ple.ple_embedding.
   ngram_embedding.shard_{0..127}.weight`, entirely inside
   `model-00001-of-00004.safetensors` (which contains nothing else). At
   `--tensor-parallel-size 1` there's no parallelism to shard this across,
   so it's one 95.37GiB tensor — more than fits alongside everything else
   on a ~95GB GPU (everything else is ~75.7GB). Fixed with:
   `--cpu-offload-gb 100 --cpu-offload-params ngram_embedding
   --offload-backend uva` (segment-matches the one parameter-name segment
   shared by all 128 shards; UVA is a straight name/byte-budget match,
   simpler than the "prefetch" backend's layer-group offloading for a
   table that lives in exactly one specific layer).
   The model card's own recipe (`https://recipes.vllm.ai/Qwen/
   Qwen3.8-Flash-Next`) claims a `VLLM_PLE_CPU_OFFLOAD=1` env var for this
   — it does not exist in vLLM 0.29.0, 0.30.0, or this nightly build
   (checked all three); that part of the recipe is inaccurate. The generic
   `OffloadConfig` flags above are the real mechanism, and only work
   because this nightly build's offloader converts the embedding *before*
   materializing it on GPU — stable 0.29.0/0.30.0's offloader still built
   the full 95.37GiB tensor on GPU first and OOM'd regardless of these
   flags.

3. **KV cache also needs CPU offload, and the `expandable_segments`
   allocator setting must be dropped.** After the fix above, weight
   loading succeeds (~74-76GiB GPU-resident), leaving only ~17-19GiB free
   — not enough for vLLM's real KV-cache pool at any reasonable
   concurrency. Fixed with vLLM's `OffloadingConnector` KV connector:
   `--kv-transfer-config '{"kv_connector":"OffloadingConnector",
   "kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":
   34359738368,"blocks_per_chunk":4}}'`. `cpu_bytes_to_use` is 32GiB, not
   the 512GiB first considered — this box has 502GB total RAM and was
   measured at ~187-190GiB *available* with the ngram-table offload above
   already active; 512GiB doesn't fit under any measurement taken, and
   this is a shared host. This connector is also incompatible with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (carried over from
   `qwen3.8-27b`'s env originally) — vLLM's config validation rejects the
   combination outright (PyTorch's CUDA VMM allocator can remap KV-cache
   virtual addresses, invalidating offloaded/pinned KV memory). Dropped
   that env var; no evidence this model actually needs it.

4. **CUDA-graph capture needs `--max-num-seqs` kept low, not
   `--max-model-len`.** vLLM's CUDA-graph-capture memory-profiling step
   (`_init_minimal_kv_cache_for_profiling` in
   `vllm/v1/worker/gpu/cudagraph_utils.py`) always tried to allocate a
   fixed ~18.95GiB "minimal" KV cache regardless of `--max-model-len`
   (tested 8192 and 4096 — identical failure both times, ~17.45-18.65GiB
   free after weights). Reading that function directly: its size is
   `min(max_num_seqs, max_cudagraph_capture_size)` — driven by
   `--max-num-seqs`, not context length. `--enforce-eager` (skip CUDA
   graphs entirely) was the first fix and got the model running, but cost
   ~5-12x slower decode than `qwen3.8-27b` on this hardware (see
   `bench_results/eager_depth*.json`). Dropping `--max-num-seqs` to 4 (then
   confirmed working at 8 too, same speed, no downside — see
   `bench_results/cudagraph_maxseqs8_depth1024.json`) let CUDA graphs
   capture successfully instead, which is the config actually running now.
   With CUDA graphs: **~10.9ms/tok decode (TPOT/ITL), ~104ms TTFT @
   depth=1024** — actually *faster* than `qwen3.8-27b`'s own no-spec
   baseline (16-18ms/tok) on the same hardware.

5. **MTP speculative decoding: tried, rejected.**
   `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
   loads and runs (46% acceptance rate, better than the model card's own
   ~36% on H100) — but made real per-token latency *worse*, not better:
   ITL (actual wall-clock gap between streamed tokens — what a user
   perceives) went from 10.88ms to 17.84ms, a ~64% regression, despite
   TPOT looking better (7.62ms) because TPOT divides compute time across
   *accepted* tokens per verification round rather than measuring real
   inter-token gaps. This matches the model card's own H100 finding,
   transposed to this GPU. **Not enabled in the final config.** (Separately,
   `vllm/model_executor/models/config.py`'s Qwen4Exp validation also
   accepts `"ngram"`/`"ngram_gpu"` as speculative-decoding methods — vLLM's
   generic prompt-lookup proposers, unrelated to the PLE `ngram_embedding`
   table above despite the name collision. Not tested.)

Final flags (see `serve.sh`): `--tensor-parallel-size 1
--gpu-memory-utilization 0.95 --max-model-len 262144 --max-num-seqs 8
--cpu-offload-gb 100 --cpu-offload-params ngram_embedding
--offload-backend uva --kv-transfer-config {...OffloadingConnector...}`, no
`--enforce-eager`, no `--speculative-config`.

## Still open / unverified

- **Tool-call / reasoning parsers.** `--tool-call-parser qwen3_coder` and
  `--reasoning-parser qwen3` are still the `qwen3.8-27b` guesses, carried
  over without verifying against this checkpoint's own `chat_template.
  jinja` (a quick grep found no obvious `qwen3_coder`-style markers).
  Reasoning-content extraction itself works in practice (confirmed via a
  real `/v1/chat/completions` request — `reasoning` and `content` split
  correctly), but tool-calling specifically hasn't been exercised.
- **Vision flags.** Not exercised — this checkpoint is vision-capable
  (`image_token_id`, `vision_config` present) but no multimodal request has
  been sent. At `--tensor-parallel-size 1`, `--mm-encoder-tp-mode` doesn't
  apply (it only matters when splitting the vision tower across TP ranks).
- **Long context.** `--max-model-len 262144` (native) is set, but only
  tested up to depth=8192 in benchmarks. Not confirmed whether the
  `OffloadingConnector`'s host-RAM KV capacity is actually reachable for a
  single very-long-context request, or only the GPU-resident pool.
- **`--max-num-seqs` ceiling.** Tested at 4 and 8, both fine, same speed.
  Not pushed further to find the actual ceiling before CUDA-graph capture
  OOMs again.
