# CLAUDE.md — qwen3.8_hosting

Repo-local context that is expensive to rediscover. Focus: **request routing / GPU
admission topology**, because the failure modes here are not visible from this repo alone.

(`AGENTS.md` is a symlink to this file so both Claude Code and pi load the same context.
Keep editing `CLAUDE.md`.)

---

## The one-paragraph mental model

There are **two machines** in the request path, and this repo only controls one of them.

```
pi (this box)   PI_MODEL=qwen3.8-flash-next-bfg, PI_PROVIDER=litellm
  └─ http://localhost:4000/v1        ← SSH tunnel to the LiteLLM proxy on another machine.
       └─ LiteLLM proxy (remote host)
            │  api_base http://localhost:12002/v1 is resolved **on the gateway host**, where
            │  it is the far end of a tunnel coming back to this box.
            └─ http://localhost:12000/v1   ← where the tunnel lands on THIS box. Under llama-swap
                                             that was the server; with the always-on service
                                             (vLLM on :8000) the tunnel must target :8000.
```

**Verified end-to-end 2026-10-07**: a request for `qwen3.8-flash-next-bfg` through `:4000`
returned 200 with `system_fingerprint: vllm-…`. `api_base` is resolved on the gateway host
(where `:12002` is a forwarded port that lands on this box's `:12000`), so nothing here needs "repointing".

This branch (`flash-next-only`) serves **only** flash-next; `qwen3.8-27b` was removed from
the repo. The gateway may still list `qwen3.8-27b-bfg` — that is gateway-side config and
will fail once nothing serves `qwen3.8-27b` here.

## Port map (verified 2026-10-07 via `ss -tlnp` + LiteLLM `/model/info`)

| Port | Bound on | What it actually is | Owner |
|---|---|---|---|
| `4000` | `127.0.0.1` + `::1` | **SSH tunnel** → LiteLLM proxy on another machine. No local process, no Docker container binds it; its socket inode maps to no FD readable by uid 1000. | remote (tunnel) |
| `8000` | `*` | **vLLM** (`bfg-flash-next.service`, `serve.sh`). Something else already held `:8000` when this was written — check `ss -tlnp` before starting. | this repo |
| `12000` | `*` | Where the gateway tunnel lands. Was llama-swap (`bfg-llama-swap.service`); **nothing listens here after cutover** until the tunnel is repointed to `:8000`. | remote (tunnel) |
| `12002` **on the gateway host** | *(not this box)* | The gateway-side end of the tunnel that reaches this box's `:12000`. **Nothing needs to listen on `12002` here** — `ss` finding no `:12002` listener on this box is expected, not a fault. (`model.env` used to say `PORT=12002`; that was a coincidence and is now `13000`.) | remote (tunnel) |
| `12001` | `127.0.0.1` | `bfg-probe` (`~/Projects/05_setup/bfg_probe/probe_server.py`, `bfg-probe.service`). **Unrelated fixed service.** Do not run vLLM on it — llama-swap once collided with it. | this repo / richard |
| `12010` | — | Another host's dev vLLM, referenced by gateway models named `*-dev-vllm`. Not this box. | remote |

## How to re-verify (cheap, read-only)

```bash
ss -tlnp | grep -E ':(4000|8000|12000|12001|12002)\b'
curl -s localhost:4000/health/readiness            # {"status":"healthy","db":"connected"} = LiteLLM
curl -s localhost:8000/v1/models                   # the vLLM server
curl -s -H "Authorization: Bearer $LITELLM_KEY" localhost:4000/model/info \
  | jq -r '.data[] | [.model_name, .litellm_params.api_base] | @tsv'
```

`$LITELLM_KEY` (25 chars) is in the environment; `~/.pi/agent/models.json` defines the
`litellm` provider (`baseUrl: http://localhost:4000/v1`) with models
`qwen3.8-27b-bfg`, `qwen3.8-flash-next-bfg`, `qwen3.8-27b-spark` (only `-flash-next-bfg` is
served from this repo).

**The gateway is reachable but its config/logs are not on this box.** Do not go hunting
for `litellm_config.yaml` here — it is not on this host, and there is no litellm proxy
container (only `litellm-db`, postgres:16-alpine on `:5432`). Use `/model/info` and the
admin API over the tunnel instead.

Identifying the tunnel process locally is a **dead end, already tried**: no `ssh`/`socat`
/`autossh` process with `4000` in its cmdline, no `LocalForward 4000` in `~/.ssh/config`, and
no Docker container with a `4000` binding. The tunnel is established outside this account.

## Gateway model → upstream

`qwen3.8-flash-next-bfg` → `http://localhost:12002/v1`, upstream `model` sent:
`qwen3.8-flash-next`. Works (verified). **Do not "fix" its `api_base`.** Check the route by
asking it:

```bash
curl -s localhost:4000/v1/chat/completions -H "Authorization: Bearer $LITELLM_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-bfg","messages":[{"role":"user","content":"ping"}],"max_tokens":8}'
```

A `system_fingerprint` beginning `vllm-` proves a real backend was reached. And check
`env | grep PI_MODEL` first — if the current session is already running on a gateway model,
the gateway is by definition working.

## The `503 "No host has sufficient memory"` incident

Seen on the gateway when flash-next was requested while another model (`qwen3.8-27b`, now
removed from this repo) was cold-starting. The numbers in the message decoded as summed
`--gpu-memory-utilization` budgets (0.85 + 0.95 of one 97886 MiB card = 176195 MB), i.e. a
pre-check that assumed one card.

Ruled out (do not re-check): llama-swap (string absent from the binary and Go source, zero
journal hits), the stock LiteLLM package, and this repo's scripts. The emitting code lives on
the remote gateway and has never been read. Do not blame the shared `api_base`.

Direct-to-vLLM is available for isolating gateway vs backend problems: a second provider in
`~/.pi/agent/models.json` with `baseUrl: http://localhost:8000/v1` (no auth; dummy `apiKey`).

## Deployment facts worth knowing

- **Always-on, one GPU.** `bfg-flash-next.service` runs `serve.sh`, which pins
  `CUDA_VISIBLE_DEVICES` (default 0), refuses to start (`exit 1`, `REFUSING TO START`) if the
  card has less than 0.95 × 97887 MiB free, and `exec`s vllm. `Restart=on-failure` with
  `RestartSec=60` retries a busy card once a minute. No idle unload; VRAM and ~132 GB host RAM
  are held for as long as it is up.
- **Host RAM is the other limit**: `--cpu-offload-gb 100` + 32 GiB KV offload on a 502 GB
  shared box whose swap was already full. Cold start is ~3–4 min, measured solo.
- `make stop` stops the whole cgroup, so `VLLM::EngineCore` goes with it.
- **Gateway cutover is unfinished from this side**: the tunnel still lands on `:12000`; vLLM is
  on `:8000`. Repoint it (remote) or gateway requests fail.
- If a second model is ever added alongside llama-swap-style multiplexing, note llama-swap's
  default group is `swap: true, exclusive: true` and stops the other model unconditionally —
  including one still cold-starting. Declare `swap: false, exclusive: false`.
- No `expandable_segments` (vLLM rejects it with the OffloadingConnector).
- `VLLM_USE_V2_MODEL_RUNNER` defaults to `1` in `serve.sh` (matching the old llama-swap config);
  set it to `0` to disable. `bench_results/` predate it, so its effect is unmeasured.
