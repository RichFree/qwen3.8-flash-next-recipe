# BUG: llama-swap swap mode killed a cold-starting model (mutual eviction thrash)

Status: **root-caused, fix applied to `llama-swap-config.yaml`, live since 22:55:43 on
2026-10-07** (config mtime 22:51:25 < service start, so the running process loaded the
`groups:` block). Supersedes the `# No groups/exclusivity needed between the two` reasoning
that was in the config header until this fix.

## Summary

A request for `qwen3.8-flash-next` arriving while `qwen3.8-27b` was still cold-starting
**killed the loading model**, and the reverse also happened. Neither model could reach
`ready` while the other was being requested. llama-swap was not recovering from an OOM or
adjudicating memory: it was executing its normal **swap** policy, which keeps exactly one
model resident per group and stops the other one unconditionally — including a model that is
still loading weights.

The policy is correct for llama-swap's designed target (one GPU, many llama.cpp models) and
simply wrong for bfg, which has **2× RTX PRO 6000 Blackwell (97887 MiB each)** sized for one
model per GPU. llama-swap never inspects GPUs at all, so it destroyed a model that already
owned its own card while the other card sat **completely idle**.

## Timeline evidence

`journalctl --user -u bfg-llama-swap.service` (all times 2026-10-07). Request durations are
from llama-swap's own access log line, so the start time is derived by subtraction.

```
19:22:59  <qwen3.8-27b> Health check passed on :13001        27b ready
19:23–19:28  normal 200s on 27b;  GET /running -> 200 659    (one model resident)
19:29:03    request for flash-next arrives                    (~= 19:31:10 minus 2m06.9s)
19:29:06    GET /running -> 200 14                            (running list EMPTY: 27b stopped)
19:29:29    request for 27b arrives                           (~= 19:32:27 minus 2m57.9s)
19:31:10    <qwen3.8-flash-next> Health check passed          flash-next finally ready
19:31:10    [WARN] metrics skipped, HTTP status=502           its own waiting client got 502
19:31:22    <qwen3.8-flash-next> ExitError >> signal: killed  killed 12s after becoming ready
19:32:27    <qwen3.8-27b> Health check passed                 27b cold-starts (2nd time)
19:32:27    <qwen3.8-27b> recovered from client disconnection during streaming
19:32:27    POST /v1/chat/completions -> 200 0  (2m57.9s)     client got an EMPTY body
19:35:02    <qwen3.8-flash-next> Health check passed          flash-next cold-starts (2nd time)
```

**Six minutes of GPU time (19:29→19:35) and two redundant cold starts to serve two
requests**, one of which returned 502 and one of which returned a 200 with zero bytes. GPU1
was free throughout.

### Detecting this: a `/running` byte-drop is NOT diagnostic on its own

A swap eviction is **not logged**. `Stop()` transitions the process to `StateStopping`, so
`waitForCmd` only prints `process exited but not StateStopping` for *unexpected* deaths — a
clean swap kill prints nothing at all. The observable signal is the `/running` response
**size** in the access log:

```
GET /running HTTP/1.1" 200 659     <- a model is resident
GET /running HTTP/1.1" 200 14      <- nothing resident ({"running":[]})
```

The *resident* size varies with which model is up (the response embeds the whole `cmd:`
string — observed 659 / 768 / 845), so **compare against the neighbouring baseline, don't
hardcode a number**. `14` is the only fixed value: it is the empty list.

**But a drop to 14 is the normal signature of the 1200 s idle TTL too.** Counting drops is
therefore worthless — most of them are healthy behaviour. An eviction is only provable when
the unload has *no* TTL line and the timing is impossible for an idle timeout. Strongest
single discriminator: **`ExitError >> signal: killed` within ~60 s of that model's own
`Health check passed`, with no preceding `Unloading model, TTL` line** — a model nobody would
leave idle for 20 minutes inside its first minute of life.

Classified rather than counted:

```bash
journalctl --user -u bfg-llama-swap.service --since -3d --no-pager -o short-iso > /tmp/ls.log
python3 - <<'EOF'
import re, datetime as dt
events, marks = [], []
for line in open('/tmp/ls.log', errors='replace'):
    try: t = dt.datetime.fromisoformat(line.split()[0])
    except Exception: continue
    m = re.search(r'"GET /running HTTP/1\.1" 200 (\d+)', line)
    if m: events.append((t, int(m.group(1))))
    if 'Unloading model, TTL' in line: marks.append((t, 'TTL'))
    if 'ExitError >> signal: killed' in line: marks.append((t, 'KILLED'))
    if 'process exited but not StateStopping' in line: marks.append((t, 'UNEXPECTED_EXIT'))
seq = []
for t, s in events:                       # collapse consecutive equal states
    if not seq or seq[-1][1] != s: seq.append((t, s))
for i in range(1, len(seq)):
    if seq[i][1] == 14 and seq[i-1][1] > 14:
        near = sorted({k for (t, k) in marks if -15 <= (t - seq[i][0]).total_seconds() <= 40})
        mins = int((seq[i][0] - seq[i-1][0]).total_seconds() // 60)
        print(f"{seq[i][0]:%m-%d %H:%M:%S}  {mins:4d} min up  ->  {'+'.join(near) or 'UNCLASSIFIED'}")
EOF
```

The TTL check window must allow the TTL line arriving *before* the drop (the poll notices a
second or two later); a forward-only window misclassifies every legitimate idle unload.

### How often did it actually happen?

Last 3 days, 18 drops to empty — **not** 18 evictions:

| Class | Count | Basis |
|---|---|---|
| Normal idle TTL | 7 | `Unloading model, TTL of 1200s reached` adjacent to the drop |
| `make swap-restart` (22:47, 22:55 on 10-07) | 2 | llama-swap PID changes at the same instant |
| The dflash garble incident (17:59, 18:52 on 10-07) | 2 | `process exited but not StateStopping`; manual kill, see `docs/TODO.md` |
| **Provable swap evictions** | **3** | see below |
| Ambiguous | 4 | 10-06 19:28, 10-06 23:06, 10-06 20:41, 10-07 17:36 — silent or KILLED-only after 50–100 min resident; most likely TTL from a build that logged less, but not provable either way |

(The `UNCLASSIFIED` label emitted by the script is just "no adjacent marker in the window";
it is not a diagnosis. The three evictions below are separated out using the ~60 s-after-ready
discriminator and the reconstructed timeline, not by the classifier.)

The three evictions are unambiguous and each is a *different* symptom of the same mechanism:

```
10-06 23:08:47   killed 43 s after it became ready
10-07 19:29:06   killed 6 min in, mid-ping-pong (the timeline above)
10-07 19:32:57   killed 28 s after it became ready
```

So the honest characterisation: the ping-pong is **not a constant background condition** — it
is intermittent, and it only bites when two models are requested close together. It is not
rarer than it looks, because each occurrence costs two full cold starts *and* hands a 502 or
an empty 200 to whoever triggered it, which is why it was so visible when it happened.

## Root cause (source-level, `~/Projects/05_setup/llama-swap`)

1. **No `groups:` and no `matrix:` in the config** → `AddDefaultGroupToConfig`
   (`proxy/config/config.go:544`–`585`) places *every* model into one synthetic group
   `(default)` declared as `Swap: true, Exclusive: true`.
2. **`swap`/`exclusive` default to `true`** in `GroupConfig.UnmarshalYAML`
   (`proxy/config/config.go:92`–`105`). An explicitly-declared group that omits `swap: false`
   silently keeps swap mode. There is no way to get co-residence by *not* configuring groups.
3. **The kill** — `ProcessGroup.ProxyRequest` (`proxy/processgroup.go:63`–`80`):

   ```go
   if pg.swap {
       if pg.lastUsedProcess != modelID {
           if pg.lastUsedProcess != "" {
               pg.processes[pg.lastUsedProcess].Stop()   // unconditional
           }
           pg.processes[modelID].ProxyRequest(writer, request)
           pg.lastUsedProcess = modelID
       }
   }
   ```

   `lastUsedProcess` is a **single slot per group**, so one model resident per group is a
   structural property, not a memory decision.
4. **Why the loading model dies instantly rather than draining.** The fork's own
   `claude_docs/concurrent-models-findings.md` states `Stop()` waits for in-flight requests
   to drain (`process.go:434`) and that "a user mid-stream is safe across an eviction". A
   model that is still loading weights has **zero in-flight requests**, so there is nothing
   to wait for: SIGKILL lands immediately and ~2–3 min of weight loading + CUDA graph
   capture is discarded.
5. **llama-swap is not memory-aware.** `grep -riE 'nvidia|nvml' proxy/*.go` → **zero hits**.
   That findings doc says the same thing independently: the solver "never polls NVML for
   actual VRAM availability", there is "no auto-budget or queue then evict", and
   "`evict_costs` does NOT prevent eviction".

### Contributing: the config's own wrong model

The header comment (now replaced) read:

> `# No groups/exclusivity needed between the two — neither is pinned to a specific GPU`
> `# (see ${gpu_check} below), so there's no shared-resource contention forcing one to`
> `# unload the other.`

The premise is right and the conclusion is wrong: exclusivity is **not** inferred from
contention, it is the default applied unconditionally. `pick_gpu_and_launch.sh` correctly
placed the second model on the free GPU, and llama-swap then killed the first one.

## Fix applied

`llama-swap-config.yaml`:

```yaml
groups:
  bfg:
    swap: false
    exclusive: false
    members:
      - qwen3.8-27b
      - qwen3.8-flash-next
```

`make swap-validate` passes (YAML OK, llama-swap accepts it on a scratch port). **Requires
`make swap-restart`; groups are read at config load.** A restart alone, without this block,
changes nothing — that is exactly what happened when the service was restarted at 22:47
before the block existed.

Note `swap: false` does **not** make llama-swap memory-aware. It only *permits*
co-residence; whether two models may physically share the box is still decided per launch by
`${gpu_check}`, which never kills anything — it takes a GPU via non-blocking `flock` or
exits 1 with `REFUSING TO START`.

## Post-fix verification

- [x] Restart — done 22:55:43 on 2026-10-07, after the `groups:` block was written (config
      mtime 22:51:25). `swap: false` is live.
- [x] **Co-residence observed 23:12:35 on 2026-10-07.** `qwen3.8-flash-next` was already
      resident on GPU0; a single `qwen3.8-27b` request cold-started it on **GPU1** while
      flash-next was left completely untouched:

  ```
  /running -> resident count: 2
       qwen3.8-27b        ready  http://localhost:13001  ttl=1200
       qwen3.8-flash-next ready  http://localhost:13002  ttl=1200

  GPU0  94870 MiB used   (flash-next, unchanged throughout the 27b start)
  GPU1  83154 MiB used   (27b; 0.85 x 97887 = 83203 target)
  GPU0 lock: HELD        GPU1 lock: HELD
  ```

  No `ExitError`, no `not StateStopping`, no TTL line, no `/running` byte drop during the
  whole start. The vLLM PID on GPU1 has llama-swap's own PID as its parent, confirming the
  normal launcher path (`pick_gpu_and_launch.sh` → `check_gpu_and_launch.sh` → vllm).
- [x] Ordering that used to trigger the thrash (one model warm → request the other) is the
      exact case above, and it now works.

### Timing observation (contradicts an older comment)

The 27b request landed ~23:11:14, vLLM bound `:13001` at 23:12:31, health check passed
23:12:35 — **~81 s request-to-ready**, against the ~297–302 s that `healthCheckTimeout`'
comment cites. The difference is almost certainly warm page cache (that checkpoint had been
loaded repeatedly that day) plus the dflash→MTP change removing a second draft-model load.
Not a reason to lower `healthCheckTimeout`, but it means the concurrent-start concern above
should be re-measured rather than assumed from the old number.

## What the fix does NOT cover

Detailed in `CLAUDE.md` ("`pick_gpu_and_launch.sh` under `swap: false`"). Headliners:

1. **Host RAM is now the concurrent-start limit**, not VRAM: flash-next asks
   `--cpu-offload-gb 100` **plus** 32 GiB `cpu_bytes_to_use` ≈ 132 GB, and can now do it
   simultaneously with a 27b load. Box: 502 GB total / ~286 GB available / **swap 19/19 full**.
2. **`healthCheckTimeout: 600` was sized on solo cold starts** (~297–302 s measured for 27b).
   Concurrent starts contend for NVMe + page cache + pinned allocation, so each is slower
   than solo. Unvalidated; consider 900 until measured.
3. The per-GPU `flock` went from **decorative** (swap mode serialized launches, so only one
   launcher could exist) to **load-bearing** (it is now the only guard against both models
   picking GPU0). Verified sound: `VLLM::EngineCore` does **not** inherit fd 200, so an
   orphaned EngineCore releases the lock and the next launch falls back to the free-VRAM check.
4. Failure mode changed from **thrash** → **unavailability**: at 0.85/0.95 utilization on
   97887 MiB cards the ceiling is one model per GPU, so if a co-tenant holds a card the other
   model refuses to start, and llama-swap retries that launch on the next request with **no
   backoff** — "no capacity" presents as "this model is broken".

## Ruled out (do not re-investigate)

- **A memory admission check inside llama-swap** — it has no GPU code path at all.
- **`pick_gpu_and_launch.sh` refusing** — `grep -c 'REFUSING TO START'` over 3 days of
  journal: **0**. (The separate `18:01:08 <qwen3.8-27b> ExitError >> exit status 1` line is
  vLLM itself exiting 1, not the launcher.)
- **TTL** — an idle timeout logs `Unloading model, TTL of 1200s reached` before the drop, and
  it accounts for 7 of the 18 drops. The 19:29:06 and 19:32:57 kills had no such line *and*
  landed 6 min / 28 s after the model became ready, which no idle timer explains. (TTL is
  `1200`s, not the 10 min implied by commit `0fdeb80`'s subject.)
- **The remote LiteLLM gateway's `503 "No host has sufficient memory"`** — a *different*
  problem, on a different host, in a gateway-side admission hook. It refuses requests; it does
  not kill processes, so it cannot explain anything in this timeline. Note the route itself is
  **healthy**: `-bfg` requests through `:4000` reach real vLLM backends (verified
  2026-10-07 23:15). Do not "fix" the `-bfg` `api_base` — see `CLAUDE.md`. Check
  `env | grep PI_MODEL` before ever suspecting the gateway: this session may be running on it.

## Related

- `CLAUDE.md` — routing/port map, the 503 gateway issue, thrash mechanism, launcher review.
- `docs/TODO.md` — separate open issue (dflash draft-acceptance collapse → garbled output).
- Fork docs worth re-reading before touching groups/matrix:
  `~/Projects/05_setup/llama-swap/claude_docs/concurrent-models-findings.md`.
