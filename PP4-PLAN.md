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

## The split cut: what moves decode (measured)

The first lever, and the biggest one found so far. The cut points were reachable
from nothing - the search always cut evenly - so the last stage carried its layers
plus the output head while every earlier stage waited on it.

**Decode t/s by cut (same prompt, same binary; two runs each):**

| cut | stages | t/s | per-card util / watt |
|---|---|---|---|
| 12,24,36 (even) | 12/12/12/12 | 92.5 | 35/42/53/72% - 48/40/45/58 W |
| 8,20,36 | 8/12/16/12 | 97.1 / 100.5 | - |
| 8,20,40 | 8/12/20/8 | 77.1 / 80.5 | c2 73%/56 W saturated |
| **8,22,34** | **8/14/12/14** | **111.7 / 106.1, then 105.6 / 105.2** | 22/39/57/87% - 50/51/54/71 W |
| 8,20,32 | 8/12/12/16 | 109.2 / 104.8 | c3 88%/73 W |
| 6,20,34 | 6/14/14/14 | 102.1 / 96.8 | - |
| 8,24,34 | 8/16/10/14 | 99.3 / 93.6 | - |
| 8,18,34 | 8/10/16/14 | 100.6 / 96.4 | - |

**8,22,34 is the shipped cut: ~+15% decode over the even cut.**

### Why it works: expert cache coverage, not layer count

The driver is not how many layers a card runs but how many experts it can hold.
A stage's cache is what is left of its VRAM after its dense weights, so MORE
layers means FEWER experts resident, and a missed expert streams from RAM.

| cut | card | layers | cache slots / needed | coverage |
|---|---|---|---|---|
| 8,20,36 | CUDA2 | 16 | 3485 / 8192 | **43%** |
| 8,20,40 | CUDA2 | 20 | 2998 / 10240 | **29%** |
| 8,22,34 | CUDA1 | 14 | 3896 / 7168 | 54% |
| 8,22,34 | CUDA2 | 12 | 3974 / 6144 | **65%** |
| 8,22,34 | CUDA3 | 14 | 3319 / 7168 | 46% |

CUDA2 at 43% coverage measured 97-101 t/s; at 29% it collapsed to 77-80; at 65%
it reached 106-112. **Coverage is the lever, and it is bought by taking layers
AWAY from a card, not giving it more.**

### Wattage as the compute proxy

An idle stage waits on its successor and draws ~42 W; a working one 52-73 W. At
the shipped cut the four cards draw 50/51/54/71 W - no card pinned at idle and
none saturated. The bad cuts show the signature plainly: 8,20,40 put c2 at
73%/56 W (saturated) while c0 sat at 17%/42 W (waiting).

### The instrument that does not reach this path

`--exp decode_profile=1` emits its per-stage line only in the single-slot solo
window (`if (stages.size() > 0)` inside the `S == 1` branch). A `--parallel 2`
boot never takes that path, so on PP4 the per-stage phases are not available from
it; `basalt decode timing` (the aggregate) and the per-card wattage are.
