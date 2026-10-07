# TODO

## qwen3.8-27b: dflash draft acceptance collapse -> garbled output

Observed 2026-10-07, ~17:57-17:59 (bfg, llama-swap journal), on the FP8
checkpoint + dflash speculative-decoding config in `llama-swap-config.yaml`.
Output became garbled ("weird symbols") during normal usage; did not resolve
on its own and required manually killing the vLLM process (`make stop` /
`make restart`) to recover.

Journal evidence (`journalctl --user -u bfg-llama-swap.service`), `Avg Draft
acceptance rate` from the periodic SpecDecoding metrics log line:

```
17:55:51  38.4%   (healthy range all session: ~25-50%)
17:57:01  9.7%
17:57:11  2.7%    (per-position: 0.115,0.049,0.016,0.008,0,0,0)
17:57:21  41.2%   (briefly recovers)
17:57:51  2.3%
17:58:01  4.5%
17:58:11  32.5%   (recovers again)
17:58:21  37.6%
17:58:31  0.0%    (flatlines: 0.000 at every one of the 7 draft positions)
17:58:23  [WARN] error processing streaming response: no valid JSON data found in stream
17:59:01  [WARN] Process manager: force killing remaining process EngineCore
```

Mean acceptance length hit 1.00 (only the target model's own bonus token
survived, zero draft tokens accepted) right as the client-facing stream
stopped parsing as valid JSON/UTF-8 — consistent with the garbled output
reported.

At engine startup, vLLM logs a related warning every time this model
loads:

```
Using uncalibrated q_scale 1.0 and/or prob_scale 1.0 with fp8 attention.
This may cause accuracy issues.
```

This is the same class of issue the config comments for this model already
flag (SM120 + fp8 KV + dflash's masked multi-token verify path —
https://github.com/flashinfer-ai/flashinfer/issues/4731). Switching to the
explicit FP8 checkpoint + `--kv-cache-dtype fp8` (2026-09-23) was meant to
sidestep it, but this incident suggests the underlying numerical flakiness
can still surface under sustained load.

**Next steps:**
- [ ] Try to reproduce / find out if it correlates with a specific prompt,
      context length, or concurrency level, rather than just restarting next
      time it happens.
- [ ] Check whether providing calibrated q_scale/prob_scale for this
      checkpoint (instead of the uncalibrated 1.0 default) eliminates it.
- [ ] Consider whether llama-swap should auto-restart on a sustained
      near-zero draft-acceptance-rate condition, since the process doesn't
      crash on its own — it has to be killed manually.
- [ ] Watch upstream flashinfer issue #4731 for a fix.
