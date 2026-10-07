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
            └─ http://localhost:12000/v1   ← llama-swap on THIS box (gateway traffic arrives
                                             as `litellm/1.104.0` from 127.0.0.1)
                 └─ dynamic vLLM on :13001 / :13002, one per GPU
```

**Verified end-to-end 2026-10-07 23:15–23:16**: requests for `qwen3.8-flash-next-bfg` and
`qwen3.8-27b-bfg` through `:4000` both returned 200 with `system_fingerprint: vllm-…`, and the
two fingerprints **differed** — two distinct backends reached through **one** `api_base`.
That is the intended design: llama-swap multiplexes on the `model` field, so many models
legitimately share one URL. Nothing here needs "repointing".

## Port map (verified 2026-10-07 via `ss -tlnp` + LiteLLM `/model/info`)

| Port | Bound on | What it actually is | Owner |
|---|---|---|---|
| `4000` | `127.0.0.1` + `::1` | **SSH tunnel** → LiteLLM proxy on another machine. No local process, no Docker container binds it; its socket inode maps to no FD readable by uid 1000. | remote (tunnel) |
| `12000` | `*` | **llama-swap** (`bfg-llama-swap.service`, user systemd, `local_121fd93`). OpenAI-compat front door; starts/stops vLLM on demand. Serves `qwen3.8-27b`, `qwen3.8-flash-next`. | this repo / richard |
| `13001`, `13002` | `0.0.0.0` | llama-swap's **dynamic** vLLM backends (startPort 13001+). Port changes per start — never hardcode. | llama-swap children |
| `12002` **on the gateway host** | *(not this box)* | The gateway-side end of the tunnel that reaches this box's `:12000`. **Nothing needs to listen on `12002` here** — `ss` finding no `:12002` listener on this box is expected, not a fault. Unrelated to `models/qwen3.8-flash-next/model.env`'s old `PORT=12002`. | remote (tunnel) |
| `12001` | `127.0.0.1` | `bfg-probe` (`~/Projects/05_setup/bfg_probe/probe_server.py`, `bfg-probe.service`). **Unrelated fixed service.** Do not hand to llama-swap — it already caused a collision (noted in `llama-swap-config.yaml` and `models/qwen3.8-flash-next/model.env`). | this repo / richard |
| `12010` | — | Another host's dev vLLM, referenced by gateway models named `*-dev-vllm`. Not this box. | remote |

## How to re-verify (cheap, read-only)

```bash
ss -tlnp | grep -E ':(4000|12000|12001|12002|13001|13002)\b'
curl -s localhost:4000/health/readiness            # {"status":"healthy","db":"connected"} = LiteLLM
curl -s localhost:12000/v1/models                  # llama-swap's live model list
curl -s -H "Authorization: Bearer $LITELLM_KEY" localhost:4000/model/info \
  | jq -r '.data[] | [.model_name, .litellm_params.api_base] | @tsv'
```

`$LITELLM_KEY` (25 chars) is in the environment; `~/.pi/agent/models.json` defines the
`litellm` provider (`baseUrl: http://localhost:4000/v1`) with models
`qwen3.8-27b-bfg`, `qwen3.8-flash-next-bfg`, `qwen3.8-27b-spark`.

**The gateway is reachable but its config/logs are not on this box.** Do not go hunting
for `litellm_config.yaml` here — it is not on this host, and there is no litellm proxy
container (only `litellm-db`, postgres:16-alpine on `:5432`). Use `/model/info` and the
admin API over the tunnel instead.

Identifying the tunnel process locally is a **dead end, already tried**: no `ssh`/`socat`
/`autossh` process with `4000` in its cmdline, no `LocalForward 4000` in `~/.ssh/config`, and
no Docker container with a `4000` binding. The tunnel is established outside this account.

## Gateway model → upstream (from `/model/info`, 20 models)

16 models → `http://localhost:12000/v1`, 2 `*-dev-vllm` → `:12010`, and the two `-bfg` models
→ `http://localhost:12002/v1`.

| Gateway model | api_base | upstream `model` sent | status |
|---|---|---|---|
| `qwen3.8-27b-bfg` | `http://localhost:12002/v1` | `qwen3.8-27b` | **works** (verified) |
| `qwen3.8-flash-next-bfg` | `http://localhost:12002/v1` | `qwen3.8-flash-next` | **works** (verified) |

### Corrected: sharing one `api_base` is correct, not stale

An earlier revision of this file called the shared `:12002` "broken wiring", reasoning that
two models cannot be disambiguated on one port, and that `:12002` being unbound on this box
proved the route was dead. **Both halves were wrong:**

- `api_base` is resolved on the *gateway host*, where `:12002` is a forwarded port. This box
  never needed a `:12002` listener.
- llama-swap routes on the `model` field, not on ports. Both models arriving at one URL is
  precisely how it is meant to work.

Do **not** "fix" the `-bfg` deployments' `api_base`. To check the route, ask it:

```bash
curl -s localhost:4000/v1/chat/completions -H "Authorization: Bearer $LITELLM_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-bfg","messages":[{"role":"user","content":"ping"}],"max_tokens":8}'
```

A `system_fingerprint` beginning `vllm-` proves a real backend was reached. And check
`env | grep PI_MODEL` first — if the current session is already running on a gateway model,
the gateway is by definition working.

## The `503 "No host has sufficient memory"` incident (qwen3.8-flash-next vs qwen3.8-27b)

Observed while starting flash-next while 27b was mid-cold-start.

The numbers decode cleanly as **`--gpu-memory-utilization` budgets, not measured usage**,
against one card's ceiling (RTX PRO 6000 Blackwell, `memory.total` = 97887 MiB):

```
 83203 MB ≈ 0.85 × 97886   qwen3.8-27b        (--gpu-memory-utilization 0.85)
 92992 MB ≈ 0.95 × 97886   qwen3.8-flash-next (--gpu-memory-utilization 0.95)
 83203 + 92992 = 176195    ← exactly the "peak" in the error
```

So the component that rejected the request **summed both models' reserved budgets and
compared them to one card's ceiling** — it refused the second without checking whether a
second physical GPU was free. This box has two cards, and `pick_gpu_and_launch.sh` would have
placed them on GPU0/GPU1 (verified doing exactly that later the same evening).

What is **not** established is *why* it assumed one card. Do not blame the shared `api_base`
— that is the correct configuration (see "Corrected: sharing one `api_base`" above). This is a
gateway-side admission rule whose basis is simply unknown.

### Ruled out (do not re-check these — already eliminated)

- **llama-swap** — the string does not exist in the `local_121fd93` binary (`strings`) nor in
  the Go source fork at `~/Projects/05_setup/llama-swap`. Zero matches in
  `journalctl --user -u bfg-llama-swap.service` across all restart history for
  `No host has sufficient memory`, `peak`, `92992`, `176195`.
- **stock LiteLLM python package** — not in the installed `~/.venv` package source.
- **this repo's own scripts** (`pick_gpu_and_launch.sh`, `check_gpu_and_launch.sh`,
  `serve*.sh`, `llama-swap-config.yaml`) — grepped, no such message.

### Still unproven

The code emitting that 503 lives **on the remote gateway host** (a custom hook / admission
check in its LiteLLM config) and has never been read. The budget-summing behaviour above is
inference from the numbers alone. What *is* verified is that the route itself is healthy, so
this is an over-restrictive pre-check rather than a wiring fault — and it fires *before*
llama-swap is consulted at all, which is why nothing related ever appears in the llama-swap
journal.

## The thrash: llama-swap only ever keeps ONE model resident (source-verified)

Full write-up with journal evidence, the `/running` byte-count tripwire, and a verification
checklist: **`docs/bug-llama-swap-swap-mode-thrash.md`**. Summary here.

Symptom: a request for model B while A is cold-starting **kills A**, B starts, and nothing
gets served. Alternating A/B requests thrash indefinitely (each cold start is 3–5 min).

This is **not** a memory heuristic. `grep -riE 'nvidia|nvml' proxy/*.go` in
`~/Projects/05_setup/llama-swap` returns **zero hits** — llama-swap never reads GPU state.
It is a bookkeeping rule in the implicit default group:

1. `AddDefaultGroupToConfig` (`proxy/config/config.go:544`–`585`): if the config declares
   **no `groups:` and no `matrix:`** — bfg's case — *every* model is dumped into one
   synthetic group `(default)` with `Swap: true, Exclusive: true`.
2. `GroupConfig.UnmarshalYAML` (`proxy/config/config.go:92`–`105`): `swap` and `exclusive`
   **default to `true`** even for explicitly-declared groups. Omitting `swap: false`
   silently keeps swap mode.
3. `ProcessGroup.ProxyRequest` (`proxy/processgroup.go:63`–`80`):

   ```go
   if pg.swap {
       if pg.lastUsedProcess != modelID {
           if pg.lastUsedProcess != "" {
               pg.processes[pg.lastUsedProcess].Stop()   // ← unconditional kill of the other model
           }
           pg.processes[modelID].ProxyRequest(writer, request)
           pg.lastUsedProcess = modelID
       }
   }
   ```

   One `lastUsedProcess` slot per group ⇒ **only one model per group can be resident, by
   design**, regardless of how many physical GPUs exist.
4. Why the *still-starting* model dies instantly: `Stop()` waits for in-flight requests to
   drain (documented in the fork's `claude_docs/concurrent-models-findings.md`). A model
   that is still loading weights has **zero** in-flight requests — nothing to drain — so it
   is killed immediately and ~5 min of cold-start work is thrown away.
5. Matrix/`evict_costs` do not help here. Per that same findings doc: llama-swap "never
   polls NVML for actual VRAM availability", there is "no auto-budget or queue-then-evict",
   and `"evict_costs" does NOT prevent eviction`.

So llama-swap's actual guarantee is "two models never fight over **one** GPU", achieved by
never keeping more than one resident. That is correct for its designed use case (one
llama.cpp host) and wrong for bfg: **2× RTX PRO 6000 Blackwell, 97887 MiB each**, and GPU1
was completely **idle** (26 MiB used) during the thrash. The free GPU was never considered
because llama-swap does not model GPUs at all.

Nothing in this repo kills anything. `pick_gpu_and_launch.sh` takes a **non-blocking
`flock`** on `/tmp/qwen3.8-hosting-gpu-${GPU_INDEX}.lock` (fd 200, held across `exec` into
vllm, released by the kernel on exit) and `exit 1`s with `REFUSING TO START` if no candidate
GPU has room. It correctly handed the second model GPU1 — llama-swap then destroyed the
model that already owned GPU0.

### Fix: declare both models in one non-swapping group

```yaml
groups:
  bfg:
    swap: false
    exclusive: false
    members:
      - qwen3.8-27b
      - qwen3.8-flash-next
```

Memory-safety/placement stays delegated to `pick_gpu_and_launch.sh` (per-GPU `flock` +
free-VRAM check), which is the layer that actually understands the two cards. If they can't
both fit, the loser exits 1 and llama-swap errors that request — graceful, no kill.

**The config's own header comment — "No groups/exclusivity needed between the two" — is the
incorrect assumption that caused this bug.** Do not re-add that reasoning.

If a 3rd model or multi-GPU budgeting is ever needed, this fork has a `matrix:` solver
(`proxy/matrix.go`, `docs/`, `scripts/gen_matrix.py`, example `matrix_fragment.yaml`) with
subset semantics and `evict_costs` — but it is oriented at a *single-GPU* VRAM budget and is
purely declarative (no runtime NVML), so it is not the tool for "2 GPUs, 2 big models".

## `pick_gpu_and_launch.sh` under `swap: false` (re-reviewed 2026-10-07)

Switching the group to non-swapping changes what this script is *for*. Verified facts from
the live box:

- **The flock went from decorative to load-bearing.** Under `swap: true`, llama-swap killed
  the other model before ever starting a new one, so only ONE launcher could exist at a
  time. Now two can run simultaneously, and the non-blocking per-GPU `flock` is the *only*
  thing stopping both models from choosing `GPU_CANDIDATES="0 1"` → GPU0. It holds: observed
  `GPU0 lock: HELD`, `GPU1 lock: FREE`, so a concurrent second launch deterministically lands
  on GPU1. No deadlock (non-blocking, never waits).
- **The lock does NOT leak into vLLM's children — checked.** `exec 200>` is inherited across
  the `exec` chain (same PID, intended) but `VLLM::EngineCore` (ppid = the vllm PID) has
  fd 200 **closed**. So an orphaned EngineCore releases the lock and the next launch then
  relies on the `FREE_MIB >= REQUIRED_MIB` check to avoid that card. Correct behaviour, but
  it means the lock and the VRAM can disagree — worth one deliberate ttl-orphan test.
- **Still true:** this script never kills anything. It takes a GPU or `exit 1`s with
  `REFUSING TO START`. Under `swap: false` that refusal is now the *only* protection against
  two models sharing a card. llama-swap surfaces it as an error and retries the same launch
  on the next request with **no backoff** (per the fork's findings doc), so a capacity
  shortage looks like "this model is broken", not "no room".

### New risks this creates (VRAM was never the binding constraint)

1. **Host RAM, not VRAM, is now the concurrent-start limit.** `qwen3.8-flash-next` requests
   `--cpu-offload-gb 100` **plus** `cpu_bytes_to_use=32GiB` (~132 GB host RAM), and it can now
   do that *at the same time* as a 27b cold start. Box is 502 GB total / ~286 GB available /
   **swap 19/19 GB already full**. Overlapping starts are unmeasured territory.
2. **`healthCheckTimeout: 600` was justified by *solo* measurements** (~297–302 s for 27b,
   3–4 min for flash-next — see the comment above that key). Concurrent starts contend for
   NVMe read bandwidth, page cache and pinned-memory allocation, so each is slower than solo.
   Exceeding 600 s fails the launch. Consider 900–1200 until a concurrent-start start time is
   actually measured.
3. **Fixed lock filenames in shared `/tmp`** (`/tmp/qwen3.8-hosting-gpu-N.lock`) under
   `set -euo pipefail`: if that path ever exists owned by another user, `exec 200>"$LOCK_FILE"`
   fails and the script dies outright — it does not fall through to the next candidate GPU.
   Cheap hardening: private dir (e.g. `$XDG_CACHE_HOME`) or tolerate the open failure.
4. **No candidate preference.** Both models probe GPU0 first, so first-come-gets-GPU0 — and
   GPU0 is the card bimo's own resident deployment sometimes occupies. Give one model
   `GPU_CANDIDATES="1 0"` in its `env:` block if you want stable placement.
5. **Co-residence needs both cards essentially empty.** At `0.85`/`0.95` utilization on
   97887 MiB cards, one model per GPU is the hard max. Any third-party job on a card removes
   it from the pool, and the second model then *refuses to start* rather than evicting. Right
   trade, but the failure mode changed from thrash → unavailability.

**Verified working live 2026-10-07 23:12:35**: flash-next resident on GPU0, a single 27b
request cold-started it onto GPU1, `/running` reports **2 ready**, GPU0 usage unchanged
throughout, both GPU locks HELD, zero eviction log lines. The old assumption "only 1 model
can be resident at a given time" is dead — 2 is the operating point. See
`docs/bug-llama-swap-swap-mode-thrash.md` for the evidence block.

## Fix

Nothing on this box needs changing for the 503 — the pre-check lives on the remote gateway and
is **not fixable from this repo**.

1. **Relax the gateway's pre-check** (remote-side): stop summing both models' reserved budgets
   against a single card. That is the real fix. The route is healthy — do **not** repoint
   `api_base`; the shared `:12002` → `:12000` tunnel is correct by design.
2. **Work around it** (this box, temporary): keep both models warm so the pre-check has
   nothing to refuse, or lower `--gpu-memory-utilization` so the combined budget fits one
   card's ceiling. Mitigations only.
3. **Direct-to-llama-swap exists but is rarely needed**: a second provider in
   `~/.pi/agent/models.json` with `baseUrl: http://localhost:12000/v1` (llama-swap has no
   auth, so a dummy `apiKey` works). Verified: `curl -s localhost:12000/v1/models` →
   `['qwen3.8-27b', 'qwen3.8-flash-next']`. Useful for isolating gateway problems from backend
   problems — not as a permanent bypass.

## Repo-specific gotchas

- `llama-swap-config.yaml` uses the `${gpu_check}` macro as a stand-in for the vllm binary:
  `pick_gpu_and_launch.sh` picks a GPU with room, sets `CUDA_VISIBLE_DEVICES` + `GPU_INDEX`,
  then delegates to `check_gpu_and_launch.sh` (final fit re-check + `exec` into vllm). There
  is **no** `CUDA_VISIBLE_DEVICES` in the model `env:` blocks anymore — do not re-add one.
- `cmd:` calls the launcher directly rather than `serve.sh` so llama-swap's SIGTERM lands on
  the vLLM process itself, not a wrapping shell (clean shutdown; check for orphaned vllm
  children after restarts).
- Idle TTL is **`ttl: 1200`** (20 min) for both models; llama-swap kills the process and
  frees ~85–93 GB on the target GPU after that. A cold start is expected, not a regression.
  (It was 1800 → 600 → 1200; the current committed value is 1200 — check
  `git show HEAD:llama-swap-config.yaml | grep ttl` rather than trusting commit subjects.)
- `maxconcurrentrequests`/ports: never assume `13001`/`13002`; query `:12000/v1/models` or `ss`.
