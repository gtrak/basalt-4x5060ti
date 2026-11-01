# pp4 — state at handoff

Branch `pp4` off `fork/main` (`306f81e7`), worktree `/home/gary/opt/basalt-pp4`.
Pushed to `fork/pp4`. 16 commits. CPU tests 62/63 (the one failure is the
AVX-512 test; this host has no AVX-512 — pre-existing).

## Shipped config (what is serving on :8000 now)

```
bin/basalt  --serve --model models/Qwen3.8-FN-FP4Local.basalt
  --gpu 0,1,2,3 --parallel 2 --slot-full-context --prefill 12288
  --resident-budget-gib 37 --prompt-cache 8 --spec 8 --ple-io direct
  --kv int8 --max-context 262144 --adapt-swaps 96 --resident-experts
  --decode-timing --suffix-draft 8 --grammar --short-read 512
  --layer-split 8,22,34
```

`--layer-split` implies forced mode; `BASALT_FORCE_SPLIT` no longer exists.

## Numbers

| axis | PP4 start | **PP4 now** | tp2pp2 |
|---|---|---|---|
| decode | 92.5 t/s | **102.6 - 111.7** | ~90 |
| prefill 62k | 3,545 | **3,232-3,243** | - |
| prefill 122k | 4,232 | **4,232** | 2,780 |
| expert PCIe | 0.0% | 0.0% | 55.8% |
| cache hit | 96-97% | 96-97% | 90% |

Decode is up **~11-15%**; prefill is unchanged (the cut was already near-optimal
for it, and chunk 12288 confirmed best on both prompt sizes).

## What was done

1. **Cherry-picked from tp2pp2** (topology-agnostic only): the decode per-op phase
   instrumentation, the device-stamp markers, the decode-profile gap, the per-window
   token lines, suffix-drafter tail re-seeding + `--exp draft_reseed`, `--exp
   mtp_preswap`, and a pool determinism guarantee test. The peer-pair machinery was
   deliberately left behind.
2. **`--layer-split K1,K2,..`** — the cut points were unreachable from anything; the
   search always cut evenly. Now they are an argument, and an explicit cut forces
   the split (replacing the env var).
3. **Swept the cut: 8,22,34 wins.** Nine cuts measured, decode spread 77-112 t/s.
4. **Made the four-stage search exhaustive** (was greedy for ns>=4).
5. **Flag sweep**: mtp_preswap, draft_reseed null; `host_ev` crashes the engine.

## What is known but NOT fixed

- **`predict()` picks a cut ~9% worse than the hand cut.** The exhaustive search
  returns 12,26,37 (93-97 t/s) where 8,22,34 gives 105-112. The auto cut holds MORE
  routed mass at every stage and still loses, so "coverage is the lever" is
  necessary but not sufficient. The unpriced terms are (a) CUDA0 carries **no
  expert cache at all** — pure serial latency at the head of the chain, charged the
  same `layer_ms` as a cache-carrying stage — and (b) the last stage is the wall in
  every cut (71%/54 W against 37-52% on c0-c2). Fixing this needs its own
  measurement; a wrong fix silently picks worse cuts than the heuristic.
- **Run-to-run noise is ~3%** (baseline against itself spans 105.2-111.7 t/s). A
  two-run sample cannot resolve a flag effect below that. Any future A/B needs more
  runs or a paired design.
- **`--exp host_ev=1` crashes** the engine on a long prompt read
  (`copy_i32x3_from_mapped`); `decode_events().enable(4000)` breaks that path.
- **`--exp decode_profile=1` does not reach a `--parallel 2` boot**: it emits only
  inside the single-slot solo window. On PP4 the per-stage phases come from
  `basalt decode timing` plus per-card wattage instead.
- Cache coverage is **capped by VRAM per card** (9-10.5 GiB each after dense
  weights), and RAM is fully used: the engine reports **0.0 GiB on SSD**, so there
  is no spill to reclaim and the resident budget cannot be raised (10 GB free).

## Next levers, priced

1. **Fix `predict()`'s per-stage cost** (serial position + the CUDA0 no-cache
   asymmetry) so the automatic search stops being worse than a hand cut.
2. **The head at the tail**: c3 is the wall in every cut. Splitting or moving the
   output head is the direct attack.
3. **Two windows in flight** — the ~1.8x ceiling; needs `batch <= 0`, and PP4's four
   stages are dependent *within* a window so this means two different windows.
