# pp4: a clean branch for the pipeline-parallel engine

Branched from `fork/main` (`306f81e7`, == production main) at the owner's direction:
start a clean PP4 branch, cherry-pick only what the tp2pp2 campaign proved useful,
then optimize.

## Why PP4 and not tp2pp2

Both measured on the same binary family, same prompt, same box:

| | PP4 | tp2pp2 |
|---|---|---|
| prefill, 122k fresh | **4,232 t/s** | 2,780 t/s |
| prefill, 62k fresh | 3,471-3,545 t/s | - |
| decode | **87-95 t/s** | ~90 t/s |
| decode window | **26.8-28.5 ms** | 30.1 ms |
| GPU-reach wait | **5.5-5.9 ms** | 10.6 ms |
| per-layer host | 1.2-1.6 ms | 2.8 ms |
| **expert PCIe** | **0.0%** | 55.8% routed |
| expert cache hit | 96-97% | 90% |

PP4 wins both axes. Its decode has no peer handshake and no expert wire at all: the
four cards hold the experts resident (48.9 GiB) and each stage is one card.

## What came over from tp2pp2, and what did not

Cherry-picked (topology-agnostic):

- `3944bc4a` verify: the decode window's phases, op by op, in the prefill marker mould
- `7c542dd9` verify: the decode phase markers read device stamps, not graph-captured events
- `6ce8b733` generate: the decode profile gains the gap (what the wall clock sees between windows)
- `038a137b` generate: the per-window token lines behind `--exp batch_lines`
- `5cf9facd` spec: tail re-seeding for the suffix drafter
- `635a2c1d` program: the drafter reseed behind `--exp draft_reseed`
- `f7c67efb` mtp: the next window's store swap beside the run, behind `--exp mtp_preswap`
- `30a88702` tests: the pool's expert row must not depend on the worker count

Deliberately NOT taken: the peer-pair machinery (partial-merge combine, the staging
ring, the RAM band, the band pin, the share clamp, the attention slice, the
shared-pool cadence probe). All of it exists to move work between the two cards of a
stage, which PP4 does not have. The per-op prefill timing (`9c1b4006`) was left out
too: it inserts its markers inside `PeerPrefill`'s code, which this branch lacks.

## A retraction carried over, so it is not relearned

TP2PP2-NOTES (39) claimed the decode window's 10.59 ms "GPU-reach wait" was PCIe
store-visibility latency. It is not - `ms_wait` is "how long the GPU took to reach
layer k", i.e. the GPU computing, seen from the host end. Three fixes were tried
against it (prefetch every predicted expert, spin in place, stage interleave) and all
were null, which is what the correct reading predicts.

## The decode model that survives

The window is serial by data dependency: GPU attention+router -> rings the host ->
host computes the layer's CPU experts -> publishes -> GPU runs the experts and the
next layer's attention. The handshake is not removable with a flag.

Where the headroom actually is:

1. **tokens per window** - the window cost is nearly flat in T, so raising dl from
   ~2.6 toward 5 is close to a doubling. Acceptance sets it.
2. **two windows in flight** - priced at ~1.8x on tp2pp2 and structurally impossible
   THERE (its stages are dependent within a window). PP4's four stages are also
   dependent within a window, so this needs two different WINDOWS, which is what
   `pipe_want`'s `batch <= 0` excludes.

## Measured on this branch (live)

- decode 92.7 t/s at acc 74%, window 26.83 ms, pcie 0.0%, hit 96.6%
- prefill 3,471 t/s at 62k fresh
- no expert PCIe at all

## Per-card utilization under decode (PP4, tp2pp2 binary)

card0 35.4% / 47.8 W, card1 41.9% / 40.3 W, card2 53.3% / 45.2 W, card3 72.2% / 58.0 W.

Rising monotonically: each stage waits on its successor, and stage 3 carries the
output head on top of its own 12 layers. This is the shape to attack.
