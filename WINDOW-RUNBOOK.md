# The Window Runbook — one down-period, four measurements (the night of 10-31)

Preconditions: ALL workers done (the gap worker b4e457e0 DONE + diff-reviewed CLEAN;
the rendezvous bench a1f425e6 must report its harness + per-point commands FIRST).
Workers are CPU-only — :8000 is theirs to lose, so the window opens only after the
last in-flight worker lands.

## Phase 0 — preflight (server UP)
- [ ] bench builder done; harness binary + commands in hand; diff-reviewed
- [ ] `git log --oneline -3` shows 584d6946 (the gap) on tp2pp2
- [ ] serving healthy (the last live line sane); note the current t/s

## Phase 1 — DOWN
- [ ] announce; kill the server (pid from `ss -ltnp | grep 8000`)
- [ ] verify 4 GPUs idle: `nvidia-smi --query-gpu=index,utilization.gpu,memory.used`

## Phase 2 — the rendezvous bench points (all 4 cards free, server DOWN)
- [ ] run the builder's per-point commands: {row-ship, partial-merge} x {2,4}-deep
      x {clean, jittered} + the decode-cadence mode (SM vs copy-engine)
- [ ] checksums verified in output; medians recorded to progress-tp4-rendezvous.md

## Phase 3 — boot A: the gap instrumentation
- [ ] boot: the CONTINUATION.md §7 command + `--exp decode_profile=1` (the gap term
      is in the binary; no new flags)
- [ ] SANE probe FIRST (the '!'-run regex check) — gate everything on it
- [ ] synthetic decode ~3-5 min; `dmon` in background: `nvidia-smi dmon -s puc -d 1`
- [ ] read: `grep 'basalt decode profile' <log> | grep -o 'gap [0-9.]* ms = between [0-9.]* + prep [0-9.]*' | tail -40`
      expect: between+prep==gap; wall+gap ≈ 36.5; name the 17 ms (prep vs between)
- [ ] ALSO read wall/run/commit/draft for the per-phase profile at the new binary

## Phase 4 — boot B: the share re-price (zero code)
- [ ] boot: boot A's command + `--exp qsa_lead_row_us=2.61 --exp qsa_peer_row_us=0.98`
- [ ] SANE probe; prefill A/B: the nat prompt, 3 reps, vs 2,004 t/s (same dl!)
- [ ] dmon running; per-layer metrics from the profile lines
- [ ] token-identity vs boot A on the same greedy prompt (the kill-switch)

## Phase 5 — decide + restore
- [ ] re-price wins (≥ ~5% prefill, sane, identity) → KEEP boot B's flags in the
      serve recipe (CONTINUATION.md §7 updated); else → restore boot A's exact flags
- [ ] final SANE probe on the restored config; live lines sane
- [ ] power: pull the dmon csv per phase (tp2 prefill / decode) — piggybacked, not first-class
- [ ] file everything; commit notes; push; CONTINUATION.md §2b updated

## Kill rules
- ANY degenerate output → kill, reboot boot A's config, re-probe, investigate offline
- The bench numbers only comparable at matched dl + power-labeled
- The window budget: ~45-60 min total; if the bench overruns, cut its {2}-deep arm
  (the 4-deep number is the decision)
